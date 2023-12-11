import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import mlp_classifier, rf_classifier, svm_classifier

def main():
    mainStartTime = time.time()
    results = []
    huMoments = []
    lbp = []
    modelNames = ['MLP','SVM','RF']
    print(f'[INFO] *********MLP**********.')
    results.append(mlp_classifier.main())
    print(f'[INFO] *********SVM**********.')
    results.append(svm_classifier.main())
    print(f'[INFO] *********RF**********.')
    results.append(rf_classifier.main())
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Total code execution time: {elapsedTime}s')
    for i in results:
        huMoments.append(i[0])
        lbp.append(i[1])
    plotResults(modelNames,huMoments, filename='HuMoments_all')
    plotResults(modelNames,lbp, filename='LBP_all')
    
def plotResults(modelNames,results,filename):
    fig, ax = plt.subplots()
    bar_container = ax.bar(modelNames, results,color=['red', 'green', 'blue', 'cyan'])
    ax.set_ylabel('Accuracy',weight='bold')
    ax.set_xlabel('Models',weight='bold')
    ax.set_title('Model comparison',fontsize=18,weight='bold')
    ax.bar_label(bar_container, fmt='{:,.2f}%')
    plt.savefig('./results/'+filename+datetime.now().strftime('-%d%m%Y-%H%M'), dpi=300) 
    print(f'[INFO] Plotting final results done in ./results/{getCurrentFileNameAndDateTime()}')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)

def getCurrentFileNameAndDateTime():
    fileName =  os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime    
    

if __name__ == "__main__":
    main()
