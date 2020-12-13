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
1- Read train and dev datasets. Remove any improper rows.  
2- Build tokenizer (converter from words to integers) using train and dev datasets.  
3- Convert train and dev reviews (words) to tokens (integers).  
4- Pad all reviews (with zeros from the end) so they all have same length. Padding length is equal to the length of the longest review.  
5- Build Machine Learning Model:  
  Embedding layer to be trained.  
  Bidirectional LSTM layer with 64 units and 0.2 dropout.  
  Dense layer with 256 output size, relu activation and 0.3 dropout.  
  Dense layer with 5 output size and softmax activation layer.  
  Loss function: 'sparse_categorical_crossentropy', Optimizer: 'adam', metric:'accuracy'  
#### Test time  

### Advanced Model  

## Results  

## How to run  
