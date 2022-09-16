import math
import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from DataAugmentation import DataAugmentation
import json
from HelperFunctions import readJsonContentTestData

# Used dataset macros
DefaultDatasetPath = "./dataset/NapkinData/train.json"

random.seed(27)


# DataLoader class, used for loading and saving the dataset and its attributes
# - 'test' mode:    Use only for test. It will load only the human gestures
# - 'train' mode:   Use only for training/validation. It will load only the synthetic gestures
# - 'full' mode:    Use when need to train and test at the same session. It will load everything
class DataLoader:

    # Constructor of the class
    def __init__(self, pad=True, include_fingerup=False, model_inputs='tangents_and_norm', test_size=0.2,
                 load_mode='test', isResample=False, augmentFactor=3, dataFileName=DefaultDatasetPath,
                 useSaliency=False):

        print('Starting the DataLoader construction ...')

        # Setting general data loader attributes
        self.use_padding = pad
        self.include_fingerup = include_fingerup
        self.test_size = test_size
        self.load_mode = load_mode
        self.isResample = isResample
        self.augmentFactor = augmentFactor
        self.labels_dict = {}
        self.model_inputs = model_inputs
        self.dataFileName = dataFileName
        self.useSaliency = useSaliency

        print('.. Done with attribute settings. Loading the data ...')

        # Loading train, validation and test sets
        if self.load_mode in ['train', 'validation']:
            self.train_set, self.validation_set, self.train_raw, self.validation_raw = self.__load_dataset_splitted()
            self.test_set = self.validation_set
            self.test_raw = self.validation_raw
        if self.load_mode == 'test':
            self.test_set, self.test_raw = self.__load_dataset()
            self.train_set, self.validation_set = self.test_set, self.test_set
            self.train_raw, self.validation_raw = self.test_raw, self.test_raw
        if self.useSaliency:
            self.y_saliency_train = self.get_saliency_labels(self.train_set[0])
            self.y_saliency_test = self.get_saliency_labels(self.test_set[0])

        self.y_arrowBinary_train = self.get_binary_label(self.train_set[1])
        self.y_arrowBinary_test = self.get_binary_label(self.test_set[1])

        self.y_arrowSplitMask_train, self.y_arrowSplitMask_test = self.createArrowSplitMask()

        self.raw_dataset = self.train_raw, self.validation_raw, self.test_raw
        self.raw_labels = self.train_set[1], self.validation_set[1], self.test_set[1]

        print('.. Done with data loading. Setting up classifier attributes ...')

        # Setting label repetition for the classifier
        if self.use_padding:
            self.labels_repeated = self.__get_labels_repeated()
            self.train_set_classifier = self.train_set[0], self.labels_repeated[0]
            self.validation_set_classifier = self.validation_set[0], self.labels_repeated[1]
            self.test_set_classifier = self.test_set[0], self.labels_repeated[2]

            print('.. Done with classifier attributes. Setting up regressor attributes ...')

            self.labels_regressed = self.__get_regressive_labels()
            self.train_set_regressor = self.train_set[0], self.labels_regressed[0]
            self.validation_set_regressor = self.validation_set[0], self.labels_regressed[1]
            self.test_set_regressor = self.test_set[0], self.labels_regressed[2]
        else:
            self.train_set_classifier, self.train_set_regressor = self.train_set, self.train_set
            self.validation_set_classifier, self.validation_set_regressor = self.validation_set, self.validation_set
            self.test_set_classifier, self.test_set_regressor = self.test_set, self.test_set

        self.tuple = self.train_set[0].shape[-1]

        print('Done with DataLoader construction!')

    def createArrowSplitMask(self):
        x_train, _, y_arrowSplitIndex_train = self.train_set
        x_test, _, y_arrowSplitIndex_test = self.test_set

        y_arrowSplitMask_train = np.ones((x_train.shape[0], x_train.shape[1]))
        for i in range(x_train.shape[0]):
            if y_arrowSplitIndex_train[i] == -1:
                continue
            y_arrowSplitMask_train[i, y_arrowSplitIndex_train[i] + 1:] = 0

        y_arrowSplitMask_test = np.ones((x_test.shape[0], x_test.shape[1]))
        for i in range(x_test.shape[0]):
            if y_arrowSplitIndex_test[i] == -1:
                continue
            y_arrowSplitMask_test[i, y_arrowSplitIndex_test[i] + 1:] = 0
        return y_arrowSplitMask_train, y_arrowSplitMask_test

    def get_binary_label(self, y_clf):
        y = np.zeros(y_clf.shape[0])
        for i in range(y_clf.shape[0]):
            if y_clf[i] == 5:
                y[i] = 1
        return y

    def get_saliency_labels(self, x):
        y = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(1, x.shape[1] - 1):
                if np.sum(x[i, j + 1]) == 0:  # padding
                    break
                u = x[i, j] - x[i, j - 1]
                v = x[i, j + 1] - x[i, j]
                if self.is_salient_point(u, v):
                    y[i, j] = 1
        return y

    def is_salient_point(self, u, v):
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
        angle = np.arccos(np.clip(c, -1, 1))
        if angle > np.pi / 2:
            return True
        return False

    # Function designed to load the $1 dataset
    def __load_dataset_from_file(self, filename):

        tensor_x = []
        tensor_y = []
        tensor_arrow_split_index = []
        labels_dict = {}

        jsonData = json.load(open(filename, 'r'))

        for clas, samples in jsonData.items():
            for sample in samples:
                if clas == "Arrow" and sample['arrowHeadStartIndex'] == -1 and self.load_mode != 'test':
                    continue
                if len(sample["points"]) > 75 or len(sample["points"]) < 3:
                    continue
                points = np.array(sample['points'])[:, 1:3]
                points -= points[0]  # setting origin to first point
                tensor_x.append(points)
                tensor_y.append(sample['classID'])
                tensor_arrow_split_index.append(sample['arrowHeadStartIndex'])
                labels_dict[clas] = sample['classID']

        print("Loading on {} completed.".format(filename))
        print(labels_dict)
        self.labels_dict = labels_dict

        return tensor_x, tensor_y, tensor_arrow_split_index

    # Function designed to load the $1 dataset
    def __load_dataset(self, preprocess=True):

        # Loading the dataset
        x, y, y_arrowSplitIndex = self.__load_dataset_from_file(self.dataFileName)
        x, y, y_arrowSplitIndex = np.array(x), np.array(y), np.array(y_arrowSplitIndex)

        x_raw = x

        if preprocess:
            x, y, y_arrowSplitIndex, x_raw = self.__preprocess_data(x, y, y_arrowSplitIndex)

        return (x, y, y_arrowSplitIndex), x_raw

    # Function designed to load the $1 dataset splitted in test and train sets
    def __load_dataset_splitted(self):

        (x, y, y_arrowSplitIndex), x_raw = self.__load_dataset(preprocess=False)

        # Splitting into test and training set
        x_train, y_train, y_arrowSplitIndex_train, x_test, y_test, y_arrowSplitIndex_test = \
            self.train_test_split(x, y, y_arrowSplitIndex, test_size=self.test_size)

        # Normalize x_train
        print('preprocessing train set ...')
        x_train, y_train, y_arrowSplitIndex_train, x_train_raw = \
            self.__preprocess_data(x_train, y_train, y_arrowSplitIndex_train, augment_data=True)

        # Normalize x_test
        print('preprocessing test set ...')
        x_test, y_test, y_arrowSplitIndex_test, x_test_raw = \
            self.__preprocess_data(x_test, y_test, y_arrowSplitIndex_test, augment_data=False)

        return (x_train, y_train, y_arrowSplitIndex_train), (
            x_test, y_test, y_arrowSplitIndex_test), x_train_raw, x_test_raw

    def train_test_split(self, x, y, y_arrowSplitIndex, test_size=0.2):
        total = x.shape[0]
        indices = [i for i in range(total)]
        random.shuffle(indices)
        train_size = 1 - test_size
        trainIndices = indices[:int(len(indices) * train_size)]
        testIndices = indices[int(len(indices) * train_size):]
        x_train, y_train, y_arrowSplitIndex_train = x[trainIndices], y[trainIndices], y_arrowSplitIndex[trainIndices]
        x_test, y_test, y_arrowSplitIndex_test = x[testIndices], y[testIndices], y_arrowSplitIndex[testIndices]
        return x_train, y_train, y_arrowSplitIndex_train, x_test, y_test, y_arrowSplitIndex_test

    def get_index_to_label(self):
        indexToLabel = {}
        for k, v in self.labels_dict.items():
            indexToLabel[v] = k
        return indexToLabel

    def __preprocess_data(self, x, y, y_arrowSplitIndex, augment_data=False):

        if augment_data:
            x_augment, y_augment, y_arrowSplitIndex_augment = [], [], []
            indexToLabel = self.get_index_to_label()
            dataAugmentation = DataAugmentation(augmentFactor=self.augmentFactor)
            for i in range(x.shape[0]):
                x_transformed = dataAugmentation.apply_data_transform(np.array(x[i]), className=indexToLabel[y[i]])
                x_augment.extend(x_transformed)
                y_augment.extend([y[i] for idx in range(len(x_transformed))])
                y_arrowSplitIndex_augment.extend([y_arrowSplitIndex[i] for idx in range(len(x_transformed))])
            x = np.concatenate((x, np.array(x_augment)), axis=0)
            y = np.concatenate((y, np.array(y_augment)), axis=0)
            y_arrowSplitIndex = np.concatenate((y_arrowSplitIndex, np.array(y_arrowSplitIndex_augment, dtype=int)), axis=0)

        x = [self.length_normalize_unit_vector(x[i], isResample=self.isResample, min_length=5.0, dims=self.model_inputs)
             for i in range(x.shape[0])]

        if self.include_fingerup:
            x_new = []
            for i in range(len(x)):
                fingerUpVector = np.zeros((x[i].shape[0], 1))
                fingerUpVector[-1, 0] = 1
                x_new.append(np.concatenate((x[i], fingerUpVector), axis=1))
            x = x_new
        x_raw = x
        # Padding branch
        if self.use_padding:
            x = np.array(x)
            x = tf.keras.preprocessing.sequence.pad_sequences(x, padding="post", dtype='float32')

        return x, y, y_arrowSplitIndex, x_raw

    def __get_labels_repeated(self):
        x_train, y_train, _ = self.train_set
        x_test, y_test, _ = self.test_set
        x_validation, y_validation, _ = self.validation_set

        # Converting to numpy the label lists
        y_train = np.repeat(np.array(y_train), x_train.shape[1], axis=0).reshape((y_train.shape[0], x_train.shape[1]))
        y_validation = np.repeat(np.array(y_validation), x_validation.shape[1], axis=0).reshape((y_validation.shape[0],
                                                                                                 x_validation.shape[1]))
        y_test = np.repeat(np.array(y_test), x_test.shape[1], axis=0).reshape((y_test.shape[0], x_test.shape[1]))

        return y_train, y_validation, y_test

    def __get_regressive_labels(self):
        y_train, y_validation, y_test = self.__get_labels_repeated()

        y_train = self.__transform_regression_labels(np.array(y_train, dtype='float32'), self.train_raw)
        y_validation = self.__transform_regression_labels(np.array(y_validation, dtype='float32'), self.validation_raw)
        y_test = self.__transform_regression_labels(np.array(y_test, dtype='float32'), self.test_raw)

        # Only train and validation expand their dims because they need to be feed into the model
        y_train = np.expand_dims(y_train, axis=2)
        y_validation = np.expand_dims(y_validation, axis=2)

        return y_train, y_validation, y_test

    def __transform_regression_labels(self, ys, xs):

        for i in range(ys.shape[0]):
            # Computing padding
            padded_size, unpadded_size = ys[i].shape[0], len(xs[i])
            pad_gap = padded_size - unpadded_size

            # Label transformation
            ys[i] = np.pad(np.arange(1, len(xs[i]) + 1), (0, pad_gap), 'constant', constant_values=(0, 0))
            ys[i] = np.array(ys[i] / len(xs[i]), dtype='float32')

        return ys

    def resampleEquiDistantPoints(self, points, min_length=5.0):
        points = np.array(points)
        output = [points[0]]
        for i in range(1, points.shape[0]):
            pt_prev = points[i - 1]
            pt = points[i]
            norm = np.linalg.norm(pt - pt_prev)
            while norm >= min_length:
                alpha = math.sqrt(min_length / norm)
                pt_prev = (1 - alpha) * pt_prev + alpha * pt
                output.append(pt_prev)
                norm = np.linalg.norm(pt - pt_prev)
        return output

    def length_normalize_unit_vector(self, points, min_length=5.0, isResample=False, dims='coord_and_tang'):
        # Always compute all the dims and trim at the end
        if len(points) <= 1:
            return points
        if isResample:
            points = self.resampleEquiDistantPoints(points, min_length=min_length)

        np_points = np.array(points, dtype=np.float32)
        output = [[np_points[0][0], np_points[0][1], 0.0, 0.0, 0.0]]
        for index in range(1, np_points.shape[0]):
            pt1 = np_points[index]
            pt1_prev = np_points[index - 1]
            if np.linalg.norm(pt1 - pt1_prev) == 0:
                print("two same consecutive points are found, skipping one in the output seq.")
                continue
            norm = np.linalg.norm(pt1 - pt1_prev)
            unit = (pt1 - pt1_prev) / norm
            output.append([pt1[0], pt1[1], unit[0], unit[1], norm])

        np_output = np.array(output, dtype=np.float32)

        if dims == "coord_and_tang":
            return np_output[:, 0:4]
        if dims == "coordinates":
            return np_output[:, 0:2]
        if dims == "tangents_and_norm":
            return np_output[:, 2:]
        if dims == "coord_tang_and_norm":
            return np_output

        raise RuntimeError(f"Unexpected dims value ${dims}")


if __name__ == "__main__":
    filename = 'dataset/NapkinData/train.json'
    dl = DataLoader(load_mode='train', dataFileName=filename, include_fingerup=False, model_inputs='coord_and_tang',
                    augmentFactor=0, isResample=False, useSaliency=False)
    print(dl.y_arrowSplitMask_train[dl.y_arrowBinary_train > 0][:5])
    # print(dl.y_train_arrowBinary)
    # print(dl.y_train_arrowBinary[dl.y_train_arrowBinary > 0].shape)
