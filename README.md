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

## Results  

## How to run  
