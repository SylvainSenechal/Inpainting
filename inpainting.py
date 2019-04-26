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
            if (random()*100 < percent):
                for rgb in range (3):
                    img[row][col][rgb] = 0 # TODO voir si 0 est bien pour pixel retiré, ou plutot utiliser -100
    return img

# TODO : numpy pour les forloop
def deleteRectangle(img, row, col, height, width):
    for i in range (row, row + height):
        for j in range (col, col + width):
            for rgb in range (3):
                img[i][j][rgb] = 0
    return img

def getPatch(img, row, col, h):
    patch = np.empty(shape=(h, h, 3))
    for i in range (h):
        for j in range (h):
            #print(img[row - int (h/2) + i][col - int (h/2) + j])
            patch[i][j] = img[row - int (h/2) + i][col - int (h/2) + j]
    return patch

def patchToVector(patch):
    return patch.flatten()
def vectorToPatch(vector, h):
    return vector.reshape(h, h, 3)

def getDictionary(img, h):
    dictionary = []
   
    for row in range (int (h/2), len(img) - int (h/2)):
        for col in range (int (h/2), len(img[row]) - int (h/2)):
            dictionary.append( getPatch(img, row, col, h) )
    dictionary = np.array(dictionary)
    # TODO : A opti..
    #dictionary = dictionary[np.where()]
    #dictionary = filter(lambda patch: for row in range (h): for col in range (h): for rgb in range (3): patch[row][col][rgb] != 0, dictionary)
    filteredDico = []
    for pi in range (len(dictionary)): # A Clean....
        remove = False
        patch = dictionary[pi]
        for row in range (len(patch)):
            for col in range (len(patch[row])):
                if patch[row][col][0] == 0 and patch[row][col][1] == 0 and patch[row][col][2] == 0:
                    remove = True
        if not remove:
            filteredDico.append(patch)
    return np.array(filteredDico)

def getPatchMissingPixels(img, h):
    patchMissingPx = []
   
    for row in range (int (h/2), len(img) - int (h/2)):
        for col in range (int (h/2), len(img[row]) - int (h/2)):
            patchMissingPx.append( getPatch(img, row, col, h) )
    patchMissingPx = np.array(patchMissingPx)
    # TODO : A opti..
    #dictionary = dictionary[np.where()]
    #dictionary = filter(lambda patch: for row in range (h): for col in range (h): for rgb in range (3): patch[row][col][rgb] != 0, dictionary)
    filteredMissingPx = []
    for pi in range (len(patchMissingPx)): # A Clean....
        missingPx = False
        patch = patchMissingPx[pi]
        for row in range (len(patch)):
            for col in range (len(patch[row])):
                if patch[row][col][0] == 0 and patch[row][col][1] == 0 and patch[row][col][2] == 0:
                    missingPx = True
        if missingPx:
            filteredMissingPx.append(patch)
    return np.array(filteredMissingPx)

def getMiddleDico(dico, h):
    middle = []
    for patch in range (len(dico)):
        middle.append( dico[patch][int (h/2)][int (h/2)][0] )
    return np.array(middle)

def lassoImpaiting(dictionnary, middleDictionnary):
    print("STARTING LASSO IMPAINTING")
    lasso = Lasso(alpha=0.1)
    lasso.fit(dictionnary, middleDictionnary)
    
    #predictions = lasso.predict(teX)
    #print(lasso.score(teX, teY))
    #print(lasso.intercept_)
    #print("score lassoRegression : ", score(predictions, teY), "out of : ", len(teY))
    #print(lasso.coef_)


if __name__=="__main__":
    ### PART 1 ################################################################

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
    
    ### PART 2 ################################################################
    # TODO : voir si les fonctions de noise et de manipulation d'image modifient l'image d'origine (à priori oui)
    image = readImage("akita.jpg")
    #plt.figure()
    #plt.imshow(image)
    plt.figure()
    plt.imshow(noise(image, 5))
    #plt.figure()
    #plt.imshow(deleteRectangle(image, 100, 200, 60, 120))
    
    patch = getPatch(image, 1, 2, 3)
    #print(patch)
    #print(patchToVector(patch))
    #print( vectorToPatch(patchToVector(patch), 3) )
    
    dico = getDictionary(image, 3)
    #patchMissingPx = getPatchMissingPixels(image, 3)
    middleDico = getMiddleDico(dico, 3)
    #print(middleDico)
    
    lassoImpaiting(dico, middleDico)
   
    # TODO idee : faire une fonction de dst entre imgage de base et image reconstruite pour voir la qualite
    # TODO idee : noise function avec perlin noise ?