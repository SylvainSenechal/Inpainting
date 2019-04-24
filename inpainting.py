# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from random import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def linearRegression(tX, tY, teX, teY):
    print("STARTING LINEAR REGRESSION")
    linear = LinearRegression()
    linear.fit(tX, tY)
    predictions = linear.predict(teX)

    print(linear.score(teX, teY))
    print(linear.intercept_)
    print("score linearRegression : ", score(predictions, teY), "out of : ", len(teY))
    #print(linear.coef_)
    
def ridgeRegression(tX, tY, teX, teY):
    print("STARTING RIDGE REGRESSION")
    ridge = Ridge(alpha=1.0)
    ridge.fit(tX, tY)
    predictions = ridge.predict(teX)

    print(ridge.score(teX, teY))
    print(ridge.intercept_)
    print("score ridgeRegression : ", score(predictions, teY), "out of : ", len(teY))
    #print(ridge.coef_)
    
def lassoRegression(tX, tY, teX, teY):
    print("STARTING LASSO REGRESSION")
    lasso = Lasso(alpha=0.1)
    lasso.fit(tX, tY)
    predictions = lasso.predict(teX)
    
    print(lasso.score(teX, teY))
    print(lasso.intercept_)
    print("score lassoRegression : ", score(predictions, teY), "out of : ", len(teY))
    #print(lasso.coef_)

def score(predictions, y):
    score = 0
    for i in range (len(predictions)):
        if (predictions[i] > 0 and y[i] > 0 or predictions[i] < 0 and y[i] < 0):
            score += 1
    return score

# Part 2 functions 
def readImage(path):
    image = plt.imread(path)
    # img[0] => premiere rangee de pixels
    # img[0][0] => premiere rangee et premiere colonne
    # img[0][0][0] => premiere rangee, colonne, teinte rouge
    # (255,255,255) => blanc, 0 => noir
    #print(len(image[0]))
    return np.array(image)

# TODO : numpy pour les forloop
def noise(img, percent):
    for row in range (len(img)):
        for col in range (len(image[row])):
            if (random()*100 > percent):
                for rgb in range (3):
                    img[row][col][rgb] = 0
    return img

# TODO : numpy pour les forloop
def deleteRectangle(img, row, col, height, width):
    for i in range (row, row + height):
        print(i)
        for j in range (col, col + width):
            for rgb in range (3):
                img[i][j][rgb] = 0
    return img

if __name__=="__main__":
    # Loading data
    dataTrain = load_usps("USPS_train.txt")
    trainX = dataTrain[0]
    trainY = dataTrain[1]
    dataTest = load_usps("USPS_test.txt")
    testX = dataTest[0]
    testY = dataTest[1]
    # Displaying digit
    # show_usps(trainX[0])
    
    # Creating binary dataset between digitOne and digitTwo
    # TODO : Faire une boucle de test sur toutes les digits + graph erreur
    digitOne = 3
    digitTwo = 4
    tX = []
    tY = []
    teX = []
    teY = []

    for i in range (len(dataTrain[0])):
      if trainY[i] == digitOne:
        tX.append(dataTrain[0][i])
        tY.append(1)
      if trainY[i] == digitTwo:
        tX.append(dataTrain[0][i])
        tY.append(-1)
    for i in range (len(dataTest[0])):
      if testY[i] == digitOne:
        teX.append(dataTest[0][i])
        teY.append(1)
      if testY[i] == digitTwo:
        teX.append(dataTest[0][i])
        teY.append(-1)

    # Making linearRegression between the 2 selected digits
    linearRegression(tX, tY, teX, teY)
    ridgeRegression(tX, tY, teX, teY)
    # TODO : Voir ALPHA dans lasso, plus Alpha est grand plus les résultats sont mauvais, normal ?
    lassoRegression(tX, tY, teX, teY)
    
    # VISUALISATION DU VECTEUR DE POIDS : REG.coef

#0 -> 127 -1,0, 128 => 255 : 0, 1
    
    # PART 2
    image = readImage("akita.jpg")
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(noise(image, 50))
    plt.figure()
    plt.imshow(deleteRectangle(image, 100, 200, 40, 80))