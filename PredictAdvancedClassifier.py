import sys
import pickle
import numpy as np
from SentimentClassifier import AdvancedSentimentClassifier

def predict(data_file):
    # load the save advanced classifier
    with open('advanced_classifier.pkl', 'rb') as input:
        advanced_classifier = pickle.load(input)

    # read and prepare data
    x_test = advanced_classifier.read_data(data_file, read_labels=False)
    x_test = advanced_classifier.prepare_data(x_test)

    # load trained ML model
    advanced_classifier.load_model()

    # get inference
    y_test_predicted = advanced_classifier.predict(x_test)

    # save test predicted labels
    np.savetxt("y_test_predicted_advanced.csv", y_test_predicted,fmt='%d')

if __name__ == '__main__':
    predict(*sys.argv[1:])