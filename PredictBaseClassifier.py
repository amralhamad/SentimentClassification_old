import sys
import pickle
import numpy as np
from SentimentClassifier import BaseSentimentClassifier

def predict(data_file):
    # load the save base classifier
    with open('base_classifier.pkl', 'rb') as input:
        base_classifier = pickle.load(input)

    # read and prepare data
    x_test = base_classifier.read_data(data_file, read_labels=False)
    x_test = base_classifier.prepare_data(x_test)

    # load trained ML model
    base_classifier.load_model()

    # get inference
    y_test_predicted = base_classifier.predict(x_test)

    # save test predicted labels
    np.savetxt("y_test_predicted_base.csv", y_test_predicted,fmt='%d')

if __name__ == '__main__':
    predict(*sys.argv[1:])
