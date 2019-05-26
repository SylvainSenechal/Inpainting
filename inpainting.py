# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from random import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy import sparse

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

###################################################################################################
### Part 2 functions ##############################################################################
###################################################################################################

def readImage(path):
    image = plt.imread(path)
    # img[0] => premiere rangee de pixels
    # img[0][0] => premiere rangee et premiere colonne
    # img[0][0][0] => premiere rangee, colonne, teinte rouge
    # (255,255,255) => blanc, 0 => noir
    #print(len(image[0]))
    return np.array(image)

# TODO : numpy pour les forloop
def noise(img, percent, h):
    img = np.array(img, dtype=np.int16) # Changement de type pour pouvoir affecter valeur negative aux pixels
    for row in range (int(h/2), len(img) - int(h/2)):
        for col in range (int(h/2), len(image[row]) - int(h/2)):
            if (random()*100 < percent):
                for rgb in range (3):
                    img[row][col][rgb] = -1 # TODO voir si 0 est bien pour pixel retiré, ou plutot utiliser -100
    return img

# TODO : numpy pour les forloop
def deleteRectangle(img, row, col, height, width):
    img = np.array(img, dtype=np.int16)
    for i in range (row, row + height):
        for j in range (col, col + width):
            for rgb in range (3):
                img[i][j][rgb] = -1
    return img

def getPatch(img, row, col, h):
    patch = np.empty(shape=(h, h, 3))
    for i in range (h):
        for j in range (h):
            patch[i][j] = img[row - int (h/2) + i][col - int (h/2) + j]
    return patch

def patchToVector(patch):
    return patch.flatten()
def vectorToPatch(vector, h):
    return vector.reshape(h, h, 3)

def getDictionary(img, h):
    step = h
    dictionary = []
   
    for row in range (int (h/2), len(img) - int (h/2), step):
        for col in range (int (h/2), len(img[row]) - int (h/2), step):
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
                if patch[row][col][0] == -1: #and patch[row][col][1] == -1 and patch[row][col][2] == -1:
                    remove = True
        if not remove:
            filteredDico.append(patchToVector(patch))
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
                if patch[row][col][0] == -1 and patch[row][col][1] == -1 and patch[row][col][2] == -1:
                    missingPx = True
        if missingPx:
            filteredMissingPx.append(patchToVector(patch))
    return np.array(filteredMissingPx)

def getOnePatch(img, h, nextRow):
    for row in range (nextRow, len(img) - int (h/2)):
        for col in range (int (h/2), len(img[row]) - int (h/2)):
            patch = getPatch(img, row, col, h)
            #for i in range (len(patch)):
                #for j in range (len(patch[i])):
            if patch[int (h/2)][int (h/2)][0] == -1: #and patch[int (h/2)][int (h/2)][1] == -1 and patch[int (h/2)][int (h/2)][2] == -1:
                return {"patch": patchToVector(patch), "x":row, "y":col}
    return False # Si il n'y a plus de patch à repeindre

def lassoInpainting(dictionnary, patch, h, alpha):
    print("STARTING LASSO IMPAINTING..")
    lasso = Lasso(alpha=alpha, max_iter=2000, fit_intercept=False, positive=True, selection='random', tol=0.0001)    
    
    indices = np.argwhere(patch["patch"] == -1)
    dico = np.delete(dictionnary, indices, axis=1)
    patch = np.delete(patch["patch"], indices)
    lasso.fit(dico.transpose(), patch) 

    reconstructedPatch = np.matmul(dictionnary.transpose(), lasso.coef_) # voir dico ou dictionary
    #print(patch)
    #print(reconstructedPatch)
    #print(lasso.coef_)

    return vectorToPatch(reconstructedPatch, h)

def reconstructNoisyImage(noisyImage, h, alpha = 1):
    plt.figure()
    dico = getDictionary(noisyImage, h)
    print(len(dico))
    nextRow = int(h/2) # Utilise pour ne pas recommencer la boucle en O(n^2) du début pour la recherche du prochain patch a remplir
    
    patch = getOnePatch(noisyImage, h, nextRow)
    drawPatch = 0 # Ne pas draw a chaque fois pour les perfs
    while patch != False: 
        drawPatch += 1
        nextRow = patch["x"]
        reconstructed = lassoInpainting(dico, patch, h, alpha)

        for row in range (h): # Reconstruction du patch sur les pixels manquants à partir du meilleur patch trouvé
            for col in range (h):
                if (noisyImage[row - int (h/2) + patch["x"]][col- int (h/2) + patch["y"]][0] == -1):
                    noisyImage[row - int (h/2) + patch["x"]][col- int (h/2) + patch["y"]] = reconstructed[row][col]
        #if (drawPatch%10 == 0): # Affichage régulier
        #    plt.imshow(noisyImage)
        #    plt.pause(0.05)
        patch = getOnePatch(noisyImage, h, nextRow)

    #plt.imshow(noisyImage)
    return noisyImage

def qualityReconstruction(sourceImg, reconstructedImg):
    error = 0
    for row in range (len(sourceImg)):
        for col in range (len(sourceImg[row])):
            for rgb in range (3):
                error += abs(sourceImg[row][col][rgb] - reconstructedImg[row][col][rgb])
    return error

def benchmarkAlpha(image, noisyImage):
    alphas = []
    qualities = []
    for alpha in np.arange(0, 7, 0.5):
        reconstructedImg = reconstructNoisyImage(noisyImage.copy(), 5, alpha)
        qualityRecons = qualityReconstruction(image, reconstructedImg)
        print("Quality reconstruction : ", qualityRecons)
        alphas.append(alpha)
        qualities.append(qualityRecons)
        plt.figure(3)
        plt.plot(alphas, qualities)
        
def benchmarkCoeff(image, noisyImage):
    x = []
    y = []
    dico = getDictionary(noisyImage, h)

    for alpha in np.arange(0, 1, 0.1):
        patch = getOnePatch(noisyImage, h, int(h/2))
        lasso = Lasso(alpha=alpha, max_iter=3000, fit_intercept=True, positive=True, selection='random', tol=0.0001)    
    
        indices = np.argwhere(patch["patch"] == -1)
        dico2 = np.delete(dico.copy(), indices, axis=1)
        patch = np.delete(patch["patch"], indices)
        lasso.fit(dico2.transpose(), patch)
    
        cpt = 0
        cpt2 = 0
        for i in range (len(lasso.coef_)):
            if lasso.coef_[i] > 0.01:
                cpt += 1
            if lasso.coef_[i] > 0.05:
                cpt2 += 1
        y.append(cpt)
        x.append(alpha)
        plt.figure(alpha*100)
        plt.plot(lasso.coef_)
    #plt.plot(x, y)
    plt.show()
        
if __name__=="__main__":
    #######################################################################################################################
    ### PART 1     ########################################################################################################
    #######################################################################################################################
    print("PART 1 ################################################################################################")

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
    
    
    #######################################################################################################################
    ### PART 2     ########################################################################################################
    #######################################################################################################################
    # TODO : voir si les fonctions de noise et de manipulation d'image modifient l'image d'origine (à priori oui)
    print("PART 2 ################################################################################################")
    image = readImage("autoroute.jpg")
    h = 5  # window size
    #plt.figure()
    #plt.imshow(image)
    #plt.figure()
    #plt.imshow(noise(image.copy(), 10))
    #plt.figure()
    #plt.imshow(deleteRectangle(image, 100, 200, 60, 120))
    
    #patch = getPatch(image, 1, 2, 3)
    #print(patch)
    #print(patchToVector(patch))
    #print( vectorToPatch(patchToVector(patch), 3) )
    
    #dico = getDictionary(image, 3)
    #patchMissingPx = getPatchMissingPixels(image, 3)
    #reconstructed = lassoInpainting(dico, dico[1])
    #print(reconstructed)
    
    #noisyImage = deleteRectangle(image.copy(), 70, 120, 60, 60) # akitaSmall
    #noisyImage = deleteRectangle(image.copy(), 88, 176, 17, 26) # autoroute, voiture
    #noisyImage = deleteRectangle(image.copy(), 120, 315, 20, 26) # autoroute, ligne blance
    noisyImage = noise(image.copy(), 0.05, h)
    plt.figure()
    plt.imshow(image)
    #plt.figure()
    #plt.imshow(noisyImage)
    #reconstructedImg = reconstructNoisyImage(noisyImage, h)
    #qualityRecons = qualityReconstruction(image, reconstructedImg)
    #print("Quality reconstruction : ", qualityRecons)


    ### Benchmark
    #benchmarkAlpha(image, noisyImage)
    benchmarkCoeff(image, noisyImage)
    
    
    
    
    
    
    # %matplotlib auto
    #hsv mauvaise idee
    
    # RAPPORT A NOTER UTILITE DU ALPHA : pour avoir querlques coeff du lasso eleves et le reste a 0,
    # plutot que tous a 0.0000.. + la sparcité produit des images non flous, plus sharp => plus 
    # - Parler de la cohérance meilleure sur la bordure gauche que droite car on balaye depuis la gauche, donc
    # on reconstruit à partir des pixels qu'on possède à droite
    # TODO idee : faire une fonction de dst entre imgage de base et image reconstruite pour voir la qualite
    # TODO idee : noise function avec perlin noise ?