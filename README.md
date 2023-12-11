# Projeto Descritores e Classificadores - Covid

## Equipe
- Lucas Martins Tanaka
- Luiz Gustavo Farabello 
- Ramon Crivelari Batista

## Descritores
- [Hu Moments] - Descritor baseado no formato da imagem
- [LBP] - Descritor baseado nas texturas e histograma da imagem

## Repositório 

| Plataforma | Link |
| ------ | ------ |
| GitHub | https://github.com/Racrivelari/projetoprocessamentoimagens |


## Classificadores e acurácias
Foram utilizados 3 tipos distintos de classificadores e obtivemos 6 resultados, sendo 2 pra cada classificador, dividos entre os artefatos obtidos do descritor Hu Moments e LBP.
- [Multi Layer Perceptron - MLP] - Rede Neural, itera e analisa cada uma das entradas (histograma), resultante em uma saída
- [Support Vector Machine - SVM] - Encontrar um hiperplano de separação no espaço de características
- [Random Forest - RF] - Conjunto de árvores de decisão, classificação final obtida majoritariamente


##### Resultado utilizando Hu Moments

![Resultado 1](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/imagesREADME/huMomentsClassifierResults.png?raw=true)

##### Resultado utilizando Local Binary Patterns 

![Resultado 2](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/imagesREADME/LBPClassifierResults.png?raw=true)


## Vídeo 
![Video](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/VideoProjetoProcImagens.mp4?raw=true)

## Instruções de Uso 
- Instalar [Python] na máquina
- Instalar [Git Bash]
- Instalar [Visual Studio Code]

- Criar uma pasta local onde ficará o download do projeto
- Abrir o diretório da pasta criada e realizar o clone do projeto
```sh
cd <pasta criada pro projeto>
git clone https://github.com/Racrivelari/projetoprocessamentoimagens.git
```

- No VS Code, ir na aba de extensões e baixar as seguintes extensões:
#
![Extensão Code Runner](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/imagesREADME/codeRunner.png?raw=true)
![Extensão Python](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/imagesREADME/python.png?raw=true)

- No próprio terminal aberto anteriormente, instale as libs do python utilizadas pela aplicação:
```sh
pip install split-folders
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install progress
pip install scikit-image
pip install matplotlib
```
- Abrir o arquivo: grayHistogram_FeatureExtraction.py e executá-lo no canto superior direito do Visual Studio Code:
#
![Exec Img](https://github.com/Racrivelari/projetoprocessamentoimagens/blob/main/imagesREADME/execPythonFileExtraction.png?raw=true)
- Após executar esse arquivo, na pasta Hu Moments Features e LBP Features surgirão novos arquivos
- Após aparecer esses arquivos, abrir o arquivo run_all_classifier.py e executá-lo da mesma maneira citada a cima
- Os resultados estarão presentes na pasta results




[//]: # (Links e Referências)

   [Multi Layer Perceptron - MLP]: <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>
   [Support Vector Machine - SVM]: <https://scikit-learn.org/stable/modules/svm.html#svm>
   [Random Forest - RF]: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>
   
   
   [Hu Moments]: <https://pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/>
   [LBP]: <https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/>
   
   [Python]: <https://www.python.org/downloads/>
   [Visual Studio Code]: <https://code.visualstudio.com/>
   [Git Bash]: <https://git-scm.com/downloads>
  
 
