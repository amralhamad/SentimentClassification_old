# SentimentClassification
--------------------------


## Used Libraries
- pandas
- numpy
- nltk
- sys
- pickle

## Used Models  
### Base Model  
#### Train time  
- Read train and dev datasets. Remove any improper rows.  
- Build tokenizer (converter from words to integers) using train and dev datasets.  
- Convert train and dev reviews (words) to tokens (integers).  
- Pad all reviews (with zeros from the end) so they all have same length. Padding length is equal to the length of the longest review.  
- Build Machine Learning Model:  
  Embedding layer to be trained.  
  Bidirectional LSTM layer with 64 units and 0.2 dropout.  
  Dense layer with 256 output size, relu activation and 0.3 dropout.  
  Dense layer with 5 output size and softmax activation layer.  
  Loss function: 'sparse_categorical_crossentropy', Optimizer: 'adam', metric:'accuracy'.
- Train the model using 256 batch size and max of 15 epachs. save_best_model and early_stopping callbacks have been used.  
- Get train and dev datasets evaluation metrics and save the predictions values in csv files.
#### Test time  
- Load base classifer object and it trained ML model.
- Read test dataset.
- Prepare dataset: tokenize and pad test dataset.
- Run model prediction over test dataset.
- Save the predictions values in a csv file.
### Advanced Model  
Inherits most of its functionalites from the base classifer with the following adding functionalites:
#### Train time  
- Convert all reviews' letters to small letters.
- Removing the "Stops Words" from all reviews. "Stop Words" are words like: 'the', 'an', ...etc. "Stop Words" values are downloaded using nltk library.
- Global Vectors for Word Representation (GloVe) has been used to build the embedding layer. Pretrained glove.840B.300d words vectors has been used for this purpose. Pretrained words vectors can be found in: https://nlp.stanford.edu/projects/glove/
- A bigger dense-layers network has been used to build the model (4 dense layers with output sizes: 256, 128, 64, 32 respectively, and 'relu' activation function and 0.2 dropout, followed by 1 5-output size softmax layer have been used). It is worth mentioning here that the advanced classifier has much less trainable parameters than the Base classifer since the embedding layer is already pretrained. 
#### Test time  
- Exactly similar functionalites to the base classifier.

## Results  
The overall accuracy over dev dataset was 70-71% using the Base classifer and 75-76% using the advanced classifer. This shows the superiority of using the advanced classifer (higher accuracy with less number of trainable parameters and faster training time). For training datasets, the overall accuracy was 91% using Base Classifier and aroung 77% using the advanced classifer. This might means that the Base classifer overfits the training dataset and this is expected since the embedding layer is trainable (so only train data has been used to train the embedding layer). The confusion matrices and the accuracies for train and dev datasets are listed below:
### Using Base Classifier
#### Train data:
################## CONFUSION MATRIX ##################  
[[6485  460&nbsp;   65&nbsp;&nbsp;   10&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;]  
&nbsp;[261&nbsp;&nbsp; 6278  465&nbsp;   19&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;]  
&nbsp;[13&nbsp;&nbsp;  219&nbsp; 6345  361&nbsp;   33&nbsp;&nbsp;]  
&nbsp;[10&nbsp;&nbsp;    9&nbsp;&nbsp;&nbsp;  211&nbsp; 6543  224&nbsp;]  
&nbsp;[1&nbsp;&nbsp;&nbsp;    2&nbsp;&nbsp;&nbsp;   14&nbsp;&nbsp;  692&nbsp; 6268]]  
################## OVERALL ACCURACY SCORE ##################  
0.9118672151754085  
#### Dev Data:  
################## CONFUSION MATRIX ##################  
[[1121  278   95    17    12]  
 [226   933   298   40    10]  
 &nbsp;[45    182   1022  207   27]  
 &nbsp;[11    21    165   1152  151]  
 &nbsp;[4     10    34    345   1093]]  
################## OVERALL ACCURACY SCORE ##################  
0.7095612748366449  
### Using Advanced Classifer
#### Train data:
################## CONFUSION MATRIX ##################  
[[6020  823  147   12   26]  
 [1106 4779 1083   44   19]  
 [ 218  620 5573  508   52]  
 [  33   70  915 5411  568]  
 [  13   14  120 1525 5305]]  
################## OVERALL ACCURACY SCORE ##################  
0.7738544166380985  
#### Dev data:
################## CONFUSION MATRIX ##################  
[[1266  204   39    6    8]  
 [ 284  947  258   10    8]  
 [  52  129 1184  110    8]  
 [   5   15  231 1106  143]  
 [   4    6   40  331 1105]]  
################## OVERALL ACCURACY SCORE ##################  
0.7478330444059208  

## How to run  
