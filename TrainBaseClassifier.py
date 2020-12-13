import sys
import pickle
import numpy as np
from SentimentClassifier import BaseSentimentClassifier

def train(train_data_file, dev_data_file, embeddings_dim =100):
    # initiate the base classifier
    base_classifier = BaseSentimentClassifier(embeddings_dim)

    # red train and dev datasets
    x_train, y_train = base_classifier.read_data(train_data_file, read_labels=True)
    x_dev, y_dev = base_classifier.read_data(dev_data_file, read_labels=True)

    # build and base classifier
    base_classifier.build_tokenizer(x_train, x_dev)

    # prepare data
    x_train = base_classifier.prepare_data(x_train)
    x_dev = base_classifier.prepare_data(x_dev)

    # build ML model and train it
    base_classifier.build_model()
    base_classifier.train_model(x_train=x_train, y_train=y_train - 1, x_dev=x_dev, y_dev=y_dev - 1)
    base_classifier.load_model()

    # get accuracies over train and dev datasets
    y_train_predicted = base_classifier.predict(x_train)
    base_classifier.get_evaluation_metrices(y_train - 1, y_train_predicted - 1)
    y_dev_predicted = base_classifier.predict(x_dev)
    base_classifier.get_evaluation_metrices(y_dev - 1, y_dev_predicted - 1)

    # save base_classifier object
    with open('base_classifier.pkl', 'wb') as output:
        pickle.dump(base_classifier, output, pickle.HIGHEST_PROTOCOL)

    # save train and dev predicted labels
    np.savetxt("y_train_predicted_base.csv", y_train_predicted,fmt='%d')
    np.savetxt("y_dev_predicted_base.csv", y_dev_predicted, fmt='%d')

if __name__ == '__main__':
    train(*sys.argv[1:])

