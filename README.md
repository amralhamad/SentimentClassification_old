# SentimentClassification
--------------------------


## Used Libraries
- pandas
- numpy
- nltk
- sys
- pickle
- sklearn

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
The overall accuracy over dev dataset was 70-71% using the Base classifer and 75-76% using the advanced classifer. This shows the superiority of using the advanced classifer (higher accuracy with less number of trainable parameters and faster training time). For training datasets, the overall accuracy was 91% using Base Classifier and aroung 77% using the advanced classifer. This might means that the Base classifer overfits the training dataset and this is expected since the embedding layer is trainable (so only train data has been used to train the embedding layer). The confusion matrices (whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and prediced label being j-th class) and the accuracies for train and dev datasets are listed below:
### Using Base Classifier
#### Train data:
################## CONFUSION MATRIX ##################  
[[6485&nbsp;&nbsp;  460&nbsp;&nbsp;&nbsp;   65&nbsp;&nbsp;&nbsp;&nbsp;   10&nbsp;&nbsp;&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[261&nbsp;&nbsp;&nbsp; 6278&nbsp;&nbsp;  465&nbsp;&nbsp;&nbsp;   19&nbsp;&nbsp;&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[13&nbsp;&nbsp;&nbsp;&nbsp;  219&nbsp;&nbsp;&nbsp; 6345&nbsp;&nbsp;  361&nbsp;&nbsp;&nbsp;   33&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[10&nbsp;&nbsp;&nbsp;&nbsp;    9&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  211&nbsp;&nbsp;&nbsp; 6543&nbsp;&nbsp;  224&nbsp;&nbsp;&nbsp;]  
&nbsp;[1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   14&nbsp;&nbsp;&nbsp;&nbsp;  692&nbsp;&nbsp;&nbsp; 6268&nbsp;&nbsp;]]  
################## OVERALL ACCURACY SCORE ##################  
0.9118672151754085  
#### Dev Data:  
################## CONFUSION MATRIX ##################  
[[1121&nbsp;&nbsp;  278&nbsp;&nbsp;&nbsp;   95&nbsp;&nbsp;&nbsp;&nbsp;    17&nbsp;&nbsp;&nbsp;&nbsp;    12&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[226&nbsp;&nbsp;&nbsp;   933&nbsp;&nbsp;&nbsp;   298&nbsp;&nbsp;&nbsp;   40&nbsp;&nbsp;&nbsp;&nbsp;    10&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[45&nbsp;&nbsp;&nbsp;&nbsp;   	182&nbsp;&nbsp;&nbsp;   1022&nbsp;&nbsp;  207&nbsp;&nbsp;&nbsp;   27&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[11&nbsp;&nbsp;&nbsp;&nbsp;    21&nbsp;&nbsp;&nbsp;&nbsp;    165&nbsp;&nbsp;&nbsp;   1152&nbsp;&nbsp;  151&nbsp;&nbsp;&nbsp;]  
&nbsp;[4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     10&nbsp;&nbsp;&nbsp;&nbsp;    34&nbsp;&nbsp;&nbsp;&nbsp;    345&nbsp;&nbsp;&nbsp;   1093&nbsp;&nbsp;]]  
################## OVERALL ACCURACY SCORE ##################  
0.7095612748366449  
### Using Advanced Classifer
#### Train data:
################## CONFUSION MATRIX ##################  
[[6020&nbsp;&nbsp;  823&nbsp;&nbsp;&nbsp;  147&nbsp;&nbsp;&nbsp;   12&nbsp;&nbsp;&nbsp;&nbsp;   26&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[1106&nbsp;&nbsp; 4779&nbsp;&nbsp; 1083&nbsp;&nbsp;   44&nbsp;&nbsp;&nbsp;&nbsp;   19&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[ 218&nbsp;&nbsp;&nbsp;  620&nbsp;&nbsp;&nbsp; 5573&nbsp;&nbsp;  508&nbsp;&nbsp;&nbsp;   52&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[  33&nbsp;&nbsp;&nbsp;&nbsp;   70&nbsp;&nbsp;&nbsp;&nbsp;  915&nbsp;&nbsp;&nbsp; 5411&nbsp;&nbsp;  568&nbsp;&nbsp;&nbsp;]  
&nbsp;[  13&nbsp;&nbsp;&nbsp;&nbsp;   14&nbsp;&nbsp;&nbsp;&nbsp;  120&nbsp;&nbsp;&nbsp; 1525&nbsp;&nbsp; 5305&nbsp;&nbsp;]]  
################## OVERALL ACCURACY SCORE ##################  
0.7738544166380985  
#### Dev data:
################## CONFUSION MATRIX ##################  
[[1266&nbsp;&nbsp;  204&nbsp;&nbsp;&nbsp;   39&nbsp;&nbsp;&nbsp;&nbsp;    6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[ 284&nbsp;&nbsp;&nbsp;  947&nbsp;&nbsp;&nbsp;  258&nbsp;&nbsp;&nbsp;   10&nbsp;&nbsp;&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[  52&nbsp;&nbsp;&nbsp;&nbsp;  129&nbsp;&nbsp;&nbsp; 1184&nbsp;&nbsp;  110&nbsp;&nbsp;&nbsp;    8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]  
&nbsp;[   5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   15&nbsp;&nbsp;&nbsp;&nbsp;  231&nbsp;&nbsp;&nbsp; 1106&nbsp;&nbsp;  143&nbsp;&nbsp;&nbsp;]  
&nbsp;[   4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   40&nbsp;&nbsp;&nbsp;&nbsp;  331&nbsp;&nbsp;&nbsp; 1105&nbsp;&nbsp;]]  
################## OVERALL ACCURACY SCORE ##################  
0.7478330444059208  
  
It is woth mentioning here that the accuracies over dev dataset for labels that have been classified correctly or with one rate off (for example 4 instead of 5) were 95.6% using Base model and 97.3% using advanced models. Results are saved in the following csv files:  
y_train_predicted_base.csv  
y_dev_predicted_base.csv  
y_test_predicted_base.csv  
y_train_predicted_advanced.csv  
y_dev_predicted_advanced.csv  
y_test_predicted_advanced.csv

## How to run  
Used base and advanced classifiers are saved in base_classifier.pkl and advanced_classifier.pkl respectively. The best ML models are saved in best_base_model.hdf5 and best_advanced_model.hdf5 for both classifiers. To retrain base of advanced classifiers, follwoing commands should be used:  

- Train Base Classifier:  
python TrainBaseClassifier.py sentiment_dataset_train.csv sentiment_dataset_dev.csv  
- Train Advanced Classifier:  
python TrainAdvancedClassifier.py sentiment_dataset_train.csv sentiment_dataset_dev.csv ./glove.840B.300d/glove.840B.300d.txt  

To use the trained classifier for prediction/inference (can be done without retraining the model), the follwing commands should be used:  
- Predict Base Classifier:  
python PredictBaseClassifier.py sentiment_dataset_test.csv   
- Predict Advanced Classifier:  
python PredictAdvancedClassifier.py sentiment_dataset_test.csv   


