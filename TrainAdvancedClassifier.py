import sys
import pickle
import numpy as np
from SentimentClassifier import AdvancedSentimentClassifier

def train(train_data_file, dev_data_file, embedding_file, embeddings_dim =100):
    # initiate the advanced classifier
    advanced_classifier = AdvancedSentimentClassifier(embeddings_dim)

    # red train and dev datasets
    x_train, y_train = advanced_classifier.read_data(train_data_file, read_labels=True)
    x_dev, y_dev = advanced_classifier.read_data(dev_data_file, read_labels=True)

    # build and advanced classifier
    advanced_classifier.build_tokenizer(x_train, x_dev)

    # prepare data
    x_train = advanced_classifier.prepare_data(x_train)
    x_dev = advanced_classifier.prepare_data(x_dev)

    # read trained embediings
    advanced_classifier.read_embedding_weights(embedding_file)

    # build ML model and train it
    advanced_classifier.build_model()
    advanced_classifier.train_model(x_train=x_train, y_train=y_train - 1, x_dev=x_dev, y_dev=y_dev - 1)
    advanced_classifier.load_model()

    # get accuracies over train and dev datasets
    y_train_predicted = advanced_classifier.predict(x_train)
    advanced_classifier.get_evaluation_metrices(y_train - 1, y_train_predicted - 1)
    y_dev_predicted = advanced_classifier.predict(x_dev)
    advanced_classifier.get_evaluation_metrices(y_dev - 1, y_dev_predicted - 1)

    # save advanced_classifier object
    with open('advanced_classifier.pkl', 'wb') as output:
        pickle.dump(advanced_classifier, output, pickle.HIGHEST_PROTOCOL)

    # save train and dev predicted labels
    np.savetxt("y_train_predicted_advanced.csv", y_train_predicted, fmt='%d')
    np.savetxt("y_dev_predicted_advanced.csv", y_dev_predicted, fmt='%d')


if __name__ == '__main__':
    train(*sys.argv[1:])

