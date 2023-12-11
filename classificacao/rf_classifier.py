import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import time
from datetime import datetime

def main():
    mainStartTime = time.time()
    trainHuMomentsFeaturePath = './HuMomentsFeatures/train/'
    testHuMomentsFeaturePath = './HuMomentsFeatures/test/'
    trainLbpFeaturePath = './LBPFeatures/train/'
    testLbpFeaturePath = './LBPFeatures/test/'
    featureFilename = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoderClasses.csv'
    print(f'[INFO] ========= TRAINING PHASE ========= ')
    trainFeatures = getFeatures(trainHuMomentsFeaturePath,featureFilename)
    trainEncodedLabels = getLabels(trainHuMomentsFeaturePath,labelFilename)
    svm1 = trainRandomForest(trainFeatures,trainEncodedLabels)
    
    trainFeatures = getFeatures(trainLbpFeaturePath,featureFilename)
    trainEncodedLabels = getLabels(trainLbpFeaturePath,labelFilename)
    svm2 = trainRandomForest(trainFeatures,trainEncodedLabels)
    print(f'[INFO] =========== TEST PHASE =========== ')
    testFeatures = getFeatures(testHuMomentsFeaturePath,featureFilename)
    testEncodedLabels1 = getLabels(testHuMomentsFeaturePath,labelFilename)
    encoderClasses1 = getEncoderClasses(testHuMomentsFeaturePath,encoderFilename)
    predictedLabels1 = predictRandomForest(svm1,testFeatures)
    
    testFeatures = getFeatures(testLbpFeaturePath,featureFilename)
    testEncodedLabels2 = getLabels(testLbpFeaturePath,labelFilename)
    encoderClasses2 = getEncoderClasses(testLbpFeaturePath,encoderFilename)
    predictedLabels2 = predictRandomForest(svm2,testFeatures)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Code execution time: {elapsedTime}s')
    accuracy1 = plotConfusionMatrix(encoderClasses1,testEncodedLabels1,predictedLabels1, namefile='HuMoments_rf')
    accuracy2 = plotConfusionMatrix(encoderClasses2,testEncodedLabels2,predictedLabels2, namefile='LBP_rf')
    return accuracy1, accuracy2
    
def getFeatures(path,filename):
    features = np.loadtxt(path+filename, delimiter=',')
    return features

def getLabels(path,filename):
    encodedLabels = np.loadtxt(path+filename, delimiter=',',dtype=int)
    return encodedLabels

def getEncoderClasses(path,filename):
    encoderClasses = np.loadtxt(path+filename, delimiter=',',dtype=str)
    return encoderClasses

    
def trainRandomForest(trainData,trainLabels):
    print('[INFO] Training the Random Forest model...')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    startTime = time.time()
    rf_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return rf_model

def predictRandomForest(rf_model,testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = rf_model.predict(testData)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels

def getCurrentFileNameAndDateTime():
    fileName =  os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime

def plotConfusionMatrix(encoderClasses,testEncodedLabels,predictedLabels, namefile):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    #Decoding test labels from numerical labels to string labels
    test = encoder.inverse_transform(testEncodedLabels)
    #Decoding predicted labels from numerical labels to string labels
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test,pred,ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: '+getCurrentFileNameAndDateTime(),fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels,predictedLabels)*100
    plt.title(f'Accuracy: {accuracy}%',fontsize=18,weight='bold')
    plt.savefig('./results/'+namefile+datetime.now().strftime('-%d%m%Y-%H%M'), dpi=300)  
    print(f'[INFO] Plotting done!')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)
    return accuracy

if __name__ == "__main__":
    main()
