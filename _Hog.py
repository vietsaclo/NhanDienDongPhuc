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

DIR_INPUT = 'Data/Train'
DIR_INPUT_TEST = 'Data/Train/testTrue'

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
            # cv2.imshow('caiLoz', img)
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
    dfs1.to_csv('./valid.csv', index = False)
    dfs2 = pd.DataFrame(arr_nonValid)
    dfs2.to_csv('./nonValid.csv', index = False)

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
    dfs1.to_csv('./valid.csv', index = False)
    dfs2 = pd.DataFrame(arr_nonValid)
    dfs2.to_csv('./nonValid.csv', index = False)

def fun_load_csv_panda(validName: str= './valid.cvs', nonValidName: str= './nonValid.csv') -> tuple:
    valid = pd.read_csv('./valid.csv')
    nonValid = pd.read_csv('./nonValid.csv')

    return valid, nonValid

def fun_getLabels(lenValid, lenNonValid)-> tuple:
    df = pd.DataFrame(columns=['lb'])
    for _ in range(lenValid):
        df = df.append({'lb': 1}, ignore_index=True)

    df1 = pd.DataFrame(columns=['lb'])
    for _ in range(lenNonValid):
        df1 = df1.append({'lb': 0}, ignore_index=True)
    
    return df, df1

def fun_train_and_save():
    valid, nonValid = fun_load_csv_panda()
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
    filename = 'Model_SVC.pkl'
    pickle.dump(svc, open(filename, 'wb'))

def fun_load_model(path: str= './Model_SVC.pkl'):
    clf = pickle.load(open(path, 'rb'))
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
    model = fun_load_model()
    for f in libs.fun_getFileNames(DIR_INPUT_TEST):
        img_path = DIR_INPUT_TEST + '/' + f
        img = fun_read_img_using_for_hog(img_path, isGray= False)
        out_hog = iv.fun_image_to_vector_myCustom(img)

        # print(out_hog)
        arr = []
        arr.append(out_hog)

        pre = model.predict(arr)
        img = cv2.resize(img, (int(720 * 0.6), int(1280 * 0.6)))
        if pre == 0:
            img = fun_putText(img, 'TH')
        else:
            img = fun_putText(img, 'CN')
        
        cv2.imshow('pre', img)
        cv2.waitKey()

if __name__ == "__main__":
    fun_test_v1()
    # fun_train_and_save()
    # fun_hog_to_csv()
    # fun_images_to_csv()
