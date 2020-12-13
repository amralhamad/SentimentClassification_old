import pandas as pd
import numpy as np
import nltk

from tensorflow.python.keras import preprocessing
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score



class BaseSentimentClassifier:
    def __init__(self, embeddings_dimension = 100):
        self.model = Sequential()
        self.embeddings_dimension = embeddings_dimension

    def read_data(self, data_file, read_labels = False):
        # read data file
        df = pd.read_csv(data_file)
        # clean data
        if read_labels:
            if df['rating'].dtype == 'object':
                df = df[df.rating.apply(lambda x: x.isnumeric())]
        # read reviews data
        x = df['review'].values.tolist()
        # read labels if available
        if read_labels:
            y = np.array(df['rating']).astype(int)
            return x, y
        else:
            return x

    def build_tokenizer(self, train_reviews, dev_reviews):
        # build vaocabulary tokenizer
        self.tokenizer = preprocessing.text.Tokenizer()
        all_reviews = train_reviews + dev_reviews
        self.tokenizer.fit_on_texts(all_reviews)
        # get longest review length and vocabulary length
        self.max_review_length = max([len(s.split()) for s in all_reviews])
        self.vocabulary_length = len(self.tokenizer.word_index) + 1

    def prepare_data(self, reviews):
        # convert reviews to tokens
        reviews_tokens = self.tokenizer.texts_to_sequences(reviews)
        # pad tokens
        tokens_padded = preprocessing.sequence.pad_sequences(reviews_tokens, maxlen=self.max_review_length,
                                                                     padding='post')
        # return padded data
        return tokens_padded

    def build_model(self):
        # build ML model
        self.model.add(Embedding(self.vocabulary_length, self.embeddings_dimension, input_length=self.max_review_length))
        self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train_model(self, x_train, y_train, x_dev, y_dev, batch_size = 256, epochs = 15, model_file_name = "best_base_model.hdf5"):
        # save best model
        checkpoint = ModelCheckpoint(model_file_name, monitor='val_accuracy', verbose = 1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=1)
        # train model
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_dev,y_dev), verbose=1, callbacks=[checkpoint, early_stopping])

    def load_model(self, model_file_name = "best_base_model.hdf5"):
        self.model = load_model(model_file_name)

    def predict(self, x_data):
        return self.model.predict_classes(x_data,batch_size=256, verbose = 1) + 1

    def get_evaluation_metrices(self, y_true, y_predicted):
        print("################## CONFUSION MATRIX ##################")
        print(confusion_matrix(y_true, y_predicted))
        print(" ")

        print("################## OVERALL ACCURACY SCORE ##################")
        print(accuracy_score(y_true, y_predicted))
        print(" ")

class AdvancedSentimentClassifier(BaseSentimentClassifier):
    def __init__(self, embeddings_dimension):
        BaseSentimentClassifier.__init__(self, embeddings_dimension)

    def read_data(self, data_file, read_labels=False):
        # read data and labels if required
        if read_labels:
            x,y = BaseSentimentClassifier.read_data(self, data_file, read_labels)
        else:
            x = BaseSentimentClassifier.read_data(self, data_file, read_labels)
        # further data processing: make all letters small case
        x = [review.lower() for review in x]
        # further processing: remove stop words
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        x_no_stops = []
        for review in x:
            review_words = review.split()
            review_words = [token for token in review_words if not token in stop_words]
            review_words = ' '.join(review_words)
            x_no_stops.append(review_words)
        if read_labels:
            return x_no_stops, y
        else:
            return x_no_stops

    def read_embedding_weights(self, embedding_file_path):
        tokens_weights = {}
        first_iteration = True
        with open(embedding_file_path, encoding="utf8") as f:
            for line in f:
                token, weights = line.split(maxsplit=1)
                weights = np.fromstring(weights, dtype="f", sep=" ")
                tokens_weights[token] = weights
                if first_iteration:
                    self.embeddings_dimension = weights.size
                    first_iteration = False
        self.embedding_weights = np.zeros((self.vocabulary_length, self.embeddings_dimension))
        for token, index in self.tokenizer.word_index.items():
            weights_vector = tokens_weights.get(token)
            if weights_vector is not None and weights_vector.size > 0:
                self.embedding_weights[index] = weights_vector

    def build_model(self):
        self.model.add(Embedding(self.vocabulary_length, self.embeddings_dimension, input_length=self.max_review_length,
                            embeddings_initializer=Constant(self.embedding_weights), trainable=False))
        self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, x_train, y_train, x_dev, y_dev, batch_size = 256, epochs = 15, model_file_name = "best_advanced_model.hdf5"):
        BaseSentimentClassifier.train_model(self, x_train, y_train, x_dev, y_dev, batch_size = batch_size, epochs = epochs, model_file_name = "advanced_base_model.hdf5")

    def load_model(self, model_file_name = "best_advanced_model.hdf5"):
        self.model = load_model(model_file_name)

