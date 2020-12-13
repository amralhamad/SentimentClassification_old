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

## How to run  
