import cv2
import numpy as np
import pandas as pd
import time
from threading import Thread

#Reading csv file with pandas and giving names to each column
def fun_readFileCSV(filePath: str= 'colors.csv'):
    index=["color","color_name","hex","R","G","B"]
    csv = pd.read_csv('colors.csv', names=index, header=None)
    return csv

#function to calculate minimum distance from all colors and get the most matching color
def fun_getColorName(csv ,R,G,B):
    minimum = 10000
    indexFind = 0
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
            indexFind = i
    return cname, indexFind

def fun_getColorName_threading(csv ,R,G,B, vector: list):
    _, index = fun_getColorName(csv, R, G, B)
    vector.append(index)
    print(index)

# function retun a vector
def fun_image_to_vector_threading(csv, img):
    vector = []
    threads = []
    if len(img) != 60 or len(img[0]) != 30:
        return None
    start = time.time()
    for i in range(len(img)):
        for j in range(len(img[i])):
            if j % 4 == 0:
                b, g, r = img[i][j]
                th = Thread(target= fun_getColorName_threading, args= (csv, r, g, b, vector))
                th.start()
                threads.append(th)
    
    # for th in threads:
    #     th.join()
    end = time.time()
    print(end-start)
    return vector

# function retun a vector
def fun_image_to_vector(csv, img):
    vector = []
    count = 0
    if len(img) != 60 or len(img[0]) != 30:
        return None
    start = time.time()
    for i in range(len(img)):
        for j in range(len(img[i])):
            if j % 4 == 0:
                b, g, r = img[i][j]
                _, index = fun_getColorName(csv, r, g, b)
                vector.append(index)
                count += 1
    end = time.time()
    print(end-start)
    return vector

if __name__ == '__main__':
    csv = fun_readFileCSV()
    img = cv2.imread('D:/imgs/OutCongNhan/cn_0.jpg')
    img = cv2.resize(img, (30, 60))
    vector = fun_image_to_vector_threading(csv, img)

