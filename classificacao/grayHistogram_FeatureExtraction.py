import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
from skimage import feature
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainHuMomentsFeaturePath = './HuMomentsFeatures/train/'
    testHuMomentsFeaturePath = './HuMomentsFeatures/test/'
    trainLbpFeaturePath = './LBPFeatures/train/'
    testLbpFeaturePath = './LBPFeatures/test/'
    
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    # HuMoments
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractGrayHistogramAndHuMomentsFeatures(trainImages)
    saveData(trainHuMomentsFeaturePath,trainEncodedLabels,trainFeatures,encoderClasses)
    
    # LBP
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractLbpFeatures(trainImages)
    saveData(trainLbpFeaturePath,trainEncodedLabels,trainFeatures,encoderClasses)
    
    print(f'[INFO] =========== TEST IMAGES =========== ')
    # HuMoments
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractGrayHistogramAndHuMomentsFeatures(testImages)
    saveData(testHuMomentsFeaturePath,testEncodedLabels,testFeatures,encoderClasses)
    
    # LBP
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractLbpFeatures(testImages)
    saveData(testLbpFeaturePath,testEncodedLabels,testFeatures,encoderClasses)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath , dirnames , filenames in os.walk(path):   
            if (len(filenames)>0): #it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}',max=len(filenames),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath,file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels,dtype=object)

def extractLbpFeatures(images):
    bar = Bar('[INFO] Extrating LBP features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if np.ndim(image) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
        hist = hist.astype("float") / hist.sum()
        featuresList.append(hist.flatten())
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)


def extractGrayHistogramAndHuMomentsFeatures(images):
    bar = Bar('[INFO] Extracting Gray histogram and Hu moments features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []

    for image in images:
        if np.ndim(image) > 2:  # > 2 = colorida
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Histograma em escala de cinza
        bins = [256]
        hist = cv2.calcHist([image], [0], None, bins, [0, 256])
        cv2.normalize(hist, hist)
        histogram_features = hist.flatten()

        # Momentos de Hu
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Concatenar as características
        combined_features = np.concatenate((histogram_features, hu_moments))

        featuresList.append(combined_features)
        bar.next()

    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels,dtype=object), encoder.classes_

def saveData(path,labels,features,encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    #the name of the arrays will be used as filenames
    #f'{labels=}' gets both variable name and its corresponding values.
    #split('=')[0] gets the variable name from f'{labels=}'
    label_filename = f'{labels=}'.split('=')[0]+'.csv'
    feature_filename = f'{features=}'.split('=')[0]+'.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0]+'.csv'
    np.savetxt(path+label_filename,labels, delimiter=',',fmt='%i')
    np.savetxt(path+feature_filename,features, delimiter=',') #float does not need format
    np.savetxt(path+encoder_filename,encoderClasses, delimiter=',',fmt='%s') 
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
