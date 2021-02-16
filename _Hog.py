import os
import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from Modules import PublicModules as libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import ImageToVector as iv
import yolov3

# Config file
DIR_INPUT = './Data/Train'
DIR_INPUT_TEST = './Data/Train/valid'
FILE_NAME_COLOR_MODEL_SVC = './Files/COLOR_MODEL_SVC.PKL'
FILE_NAME_COLOR_NON_VALID = './Files/COLOR_NON_VALID.CSV'
FILE_NAME_COLOR_VALID = './Files/COLOR_VALID.CSV'
FILE_NAME_COLOR = './Files/COLOR.CSV'
FILE_NAME_HOG_MODEL_SVC = './Files/HOG_MODEL_SVC.PKL'
FILE_NAME_HOG_NON_VALID = './Files/HOG_NON_VALID.CSV'
FILE_NAME_HOG_VALID = './Files/HOG_VALID.CSV'

def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape # 128, 64
    
    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)
    
    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx+0.00001)) # radian
    orientation = np.degrees(orientation) # -90 -> 90
    orientation += 90 # 0 -> 180
    
    num_cell_x = w // cell_size # 8
    num_cell_y = h // cell_size # 16
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins]) # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag) # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass
    
    # normalization
    redundant_cell = block_size-1
    feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
    for bx in range(num_cell_x-redundant_cell): # 7
        for by in range(num_cell_y-redundant_cell): # 15
            by_from = by
            by_to = by+block_size
            bx_from = bx
            bx_to = bx+block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten() # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any(): # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v
    
    return feature_tensor.flatten() # 3780 features

def fun_read_img_using_for_hog(img_path, isGray: bool= True):
    if isGray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
    img = cv2.resize(src=img, dsize=(64, 128))
    return img

def fun_hog_to_csv():
    dir_trains = ['Valid', 'NonValid']
    arr_valid = []
    arr_nonValid = []

    for _dir in dir_trains:
        fileNames = libs.fun_getFileNames(DIR_INPUT + '/' + _dir)
        # chay tung file va bat dau cho qua hog xac nhan
        incree = 0
        _max = len(fileNames)
        for f in fileNames:
            img_path = DIR_INPUT + '/' + _dir + '/' + f
            img = fun_read_img_using_for_hog(img_path)
            libs.fun_print_process(count= incree, max= _max)
            out_hog = hog(img)
            if _dir == 'Valid':
                arr_valid.append(out_hog)
            else:
                arr_nonValid.append(out_hog)
            
            incree += 1

        incree = 0

    print(arr_valid[0])
    print(arr_nonValid[0])

    # save file
    dfs1 = pd.DataFrame(arr_valid)
    dfs1.to_csv(FILE_NAME_HOG_VALID, index = False)
    dfs2 = pd.DataFrame(arr_nonValid)
    dfs2.to_csv(FILE_NAME_HOG_NON_VALID, index = False)

def fun_images_to_csv():
    dir_trains = ['Valid', 'NonValid']
    arr_valid = []
    arr_nonValid = []

    for _dir in dir_trains:
        fileNames = libs.fun_getFileNames(DIR_INPUT + '/' + _dir)
        # chay tung file va bat dau cho qua hog xac nhan
        incree = 0
        _max = len(fileNames)
        for f in fileNames:
            img_path = DIR_INPUT + '/' + _dir + '/' + f
            img = fun_read_img_using_for_hog(img_path, isGray= False)
            libs.fun_print_process(count= incree, max= _max)
            out_hog = iv.fun_image_to_vector_myCustom(img)
            if _dir == 'Valid':
                arr_valid.append(out_hog)
            else:
                arr_nonValid.append(out_hog)
            
            incree += 1

        incree = 0

    print(arr_valid[0])
    print(arr_nonValid[0])

    # save file
    dfs1 = pd.DataFrame(arr_valid)
    dfs1.to_csv(FILE_NAME_COLOR_VALID, index = False)
    dfs2 = pd.DataFrame(arr_nonValid)
    dfs2.to_csv(FILE_NAME_COLOR_NON_VALID, index = False)

def fun_load_csv_panda(isHog: bool= True) -> tuple:
    if isHog:
        valid = pd.read_csv(FILE_NAME_HOG_VALID)
        nonValid = pd.read_csv(FILE_NAME_HOG_NON_VALID)
    else:
        valid = pd.read_csv(FILE_NAME_COLOR_VALID)
        nonValid = pd.read_csv(FILE_NAME_COLOR_NON_VALID)

    return valid, nonValid

def fun_getLabels(lenValid, lenNonValid)-> tuple:
    df = pd.DataFrame(columns=['lb'])
    for _ in range(lenValid):
        df = df.append({'lb': 1}, ignore_index=True)

    df1 = pd.DataFrame(columns=['lb'])
    for _ in range(lenNonValid):
        df1 = df1.append({'lb': 0}, ignore_index=True)
    
    return df, df1

def fun_train_and_save(isHog: bool= True):
    valid, nonValid = fun_load_csv_panda(isHog= isHog)
    validLable, nonValidLabel = fun_getLabels(len(valid), len(nonValid))

    X = valid.append(nonValid, ignore_index=True)
    y = validLable.append(nonValidLabel, ignore_index=True)
    y = y['lb']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)

    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    svc = SVC()
    svc.fit(X_train, y_train)
    pred_svc = svc.predict(X_test)
    print(classification_report(y_test, pred_svc))
    print(confusion_matrix(y_test, pred_svc))

    #save model
    filename = FILE_NAME_HOG_MODEL_SVC if isHog else FILE_NAME_COLOR_MODEL_SVC
    pickle.dump(svc, open(filename, 'wb'))

# def fun_load_model(path: str= './Model_SVC.pkl'):
#     clf = pickle.load(open(path, 'rb'))
#     return clf
def fun_load_model(isHog: bool= True):
    if isHog:
        clf = pickle.load(open(FILE_NAME_HOG_MODEL_SVC,'rb'))
    else: 
        clf = pickle.load(open(FILE_NAME_COLOR_MODEL_SVC,'rb'))
    return clf

def fun_putText(image, mess):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    image = cv2.putText(image, mess, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
    return image

def fun_test_v1():
    model_hog = fun_load_model(isHog=False)
    model_color = fun_load_model(isHog=True)

    for f in libs.fun_getFileNames(DIR_INPUT_TEST):
        img_path = DIR_INPUT_TEST + '/' + f
        print('img_path:',img_path)
        img = fun_read_img_using_for_hog(img_path, isGray= False)
        out_hog = iv.fun_image_to_vector_myCustom(img)

        # print(out_hog)
        arr = []
        arr.append(out_hog)

        pre = model_hog.predict(arr)
        
        if pre == 1:
            img_color = fun_read_img_using_for_hog(img_path, isGray=True)
            out_hog2 = hog(img_color)
            arr1 = []
            arr1.append(out_hog2)
            pre_color = model_color.predict(arr1)

            img = cv2.resize(img_color, (int(720 * 0.6), int(1280 * 0.6)))
            
            if pre_color == 0:
                img = fun_putText(img, 'TH')
            else:
                img = fun_putText(img, 'CN')
            
            cv2.imshow('pre_color', img)
            cv2.waitKey()
        else:
            img = cv2.resize(img, (int(720 * 0.6), int(1280 * 0.6)))
            img = fun_putText(img, 'TH')
            cv2.imshow('pre_hog', img)
            cv2.waitKey()

            

        # img = cv2.resize(img, (int(720 * 0.6), int(1280 * 0.6)))
        # if pre == 0:
        #     img = fun_putText(img, 'TH')
        # else:
        #     img = fun_putText(img, 'CN')
        
        # cv2.imshow('pre', img)
        # cv2.waitKey()

def fun_test_hog():
    model_hog = fun_load_model(isHog= True)

    url = 'F:/videoTest.mp4'

    cap = cv2.VideoCapture(url)
    while True:
        isContinue, frame = cap.read()
        if not isContinue:
            break

        img = cv2.resize(src=frame, dsize=(64, 128))
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_hog = hog(img_color)

        # print(out_hog)
        arr = []
        arr.append(out_hog)

        pre = model_hog.predict(arr)

        img = cv2.resize(img, (300, 600))
        if pre[0] == 0:
            img = fun_putText(img, 'TH')
        else:
            img = fun_putText(img, 'CNNN')
        cv2.imshow('f', img)
        cv2.waitKey(10)

def fun_test_v2():
    # Load 2 model vao ram
    model_color = fun_load_model(isHog= False)
    model_hog = fun_load_model(isHog= True)

    url = 'F:/videoTest.mp4'

    cap = cv2.VideoCapture(url)
    count = 0
    while True:
        isContinue, frame = cap.read()
        if not isContinue:
            break

        img = cv2.resize(src=frame, dsize=(64, 128))
        out_hog = iv.fun_image_to_vector_myCustom(img)

        # print(out_hog)
        arr = []
        arr.append(out_hog)

        pre = model_color.predict(arr)

        status = False
        
        if pre == 1:
            print('asdfs' + str(count))
            count += 1
            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out_hog2 = hog(img_color)
            arr1 = []
            arr1.append(out_hog2)
            pre_color = model_hog.predict(arr1)

            img = cv2.resize(img_color, (int(720 * 0.6), int(1280 * 0.6)))
            
            if pre_color == 1:
                status = True

        if status:
            img = fun_putText(img, 'CNNNN')
        else:
            img = fun_putText(img, 'TH')

        cv2.imshow('f', img)
        cv2.waitKey(10)

def fun_detect_with_color(model_color, frame):
    try:
        img = cv2.resize(frame, (64, 128))
        img = iv.fun_image_to_vector_myCustom(img)

        arr = []
        arr.append(img)

        pre = model_color.predict(arr)
        return pre[0]
    except:
        return 0

def fun_detect_with_hog(model_hog, frame):
    try:
        img = cv2.resize(frame, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = hog(img)

        arr = []
        arr.append(img)

        pre = model_hog.predict(arr)
        return pre[0]
    except:
        return 0

def fun_detect_camera_readTime(urlVideo):
    # Load 2 model vao ram
    model_color = fun_load_model(isHog= False)
    model_hog = fun_load_model(isHog= True)

    url = urlVideo
    cap = cv2.VideoCapture(url)
    count = 0
    while True:
        isContinue, frame = cap.read()
        if not isContinue:
            break

        height, width, _ = frame.shape
        frame = cv2.resize(frame, (int(width * 0.7), int(height * 0.7)))
        img_show = frame.copy()
        _, imageGets = yolov3.fun_DetectObject(sourceImage= frame)
        
        # Da lay duoc ta ca nguoi
        for person in imageGets:
            status = False
            color_predict = fun_detect_with_color(model_color= model_color, frame= person[0])
            if color_predict == 1:
                print(count)
                count += 1
                hog_predict = fun_detect_with_hog(model_hog= model_hog, frame= person[0])
                if hog_predict == 1:
                    status = True
            
            if status:
                yolov3.draw_prediction(
                        img= img_show,
                        class_id= 'Digitech',
                        confidence= 'None',
                        x = person[1][2],
                        y= person[1][0],
                        x_plus_w= person[1][3],
                        y_plus_h= person[1][1]
                    )
        
        cv2.imshow('Detection', img_show)
        cv2.waitKey(1)

if __name__ == "__main__":
    # fun_test_hog()
    fun_detect_camera_readTime(0)
    # fun_test_v2()
    # fun_detect_camera_readTime(urlVideo= 0)
    # fun_train_and_save(isHog= True)
    # fun_hog_to_csv()
    # fun_images_to_csv()
