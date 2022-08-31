import time
import numpy as np
from GestuReNN_mts import GestuReNN_GRU


class GraphicManager:

    def __init__(self, dataset):
        self.dataset = dataset

    def generate_progressive_accuracy(self, model, data):

        # Basic data manipulation
        x, y = data
        # Predicting the values
        clf_pred = self.__make_predictions(model, x)
        accuracyCount,totalCount = self.computeAccuracy(clf_pred, y)
        print("class: non-arrow, accuracy: {}, total: {}".format(accuracyCount[0]/totalCount[0], totalCount[0]))
        print("class: arrow, accuracy: {}, total: {}".format(accuracyCount[1] / totalCount[1], totalCount[1]))

    def __make_predictions(self, model, x):
        clf_pred = []
        # Predicting the values
        if type(model) is GestuReNN_GRU:
            clf_pred = model.model(x)
            clf_pred = np.argmax(clf_pred, axis=1)
        else:
            print('Classifier and regressor should be instances of GestuReNN.')
            exit(1)

        return clf_pred

    def computeAccuracy(self, clf_pred, ground_truth):

        n_predictions = clf_pred.shape[0]
        accuracyCount = [0, 0]
        totalCount = [0, 0]

        for i in range(n_predictions):
            # print("--------Sample - {}----------".format(i))
            if ground_truth[i] == clf_pred[i]:
                accuracyCount[ground_truth[i]] += 1
            totalCount[ground_truth[i]] += 1
        # accuracy = accuracyCount / totalCount
        return accuracyCount,totalCount
