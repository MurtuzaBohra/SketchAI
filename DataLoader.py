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
ONE_DOLLAR_G3_SYNTH = "dataset/GGG/1$/csv-1dollar-synth-best"
ONE_DOLLAR_G3_HUMAN = "dataset/GGG/1$/csv-1dollar-human"
N_DOLLAR_G3_SYNTH = "dataset/GGG/N$/csv-ndollar-synth-best"
N_DOLLAR_G3_HUMAN = "dataset/GGG/N$/csv-ndollar-human"
Napkin_HUMAN = "/Users/murtuza/Desktop/ShapeRecognition/deep-stroke/dataset/NapkinData/csv"


# DataLoader class, used for loading and saving the dataset and its attributes
# - 'test' mode:    Use only for test. It will load only the human gestures
# - 'train' mode:   Use only for training/validation. It will load only the synthetic gestures
# - 'full' mode:    Use when need to train and test at the same session. It will load everything
class DataLoader:

    # Constructor of the class
    def __init__(self, pad=True, include_fingerup=False, model_inputs='tangents_and_norm', test_size=0.2,
                 dataset='1$', load_mode='full', augmentFactor=3, datasetFolder=Napkin_HUMAN,
                 fileType='', labelJsonPath=None, excludeClasses=['line']):

        print('Starting the DataLoader construction ...')

        # Setting general data loader attributes
        self.use_padding = pad
        self.include_fingerup = include_fingerup
        self.test_size = test_size
        self.stroke_dataset = dataset
        self.load_mode = load_mode
        self.augmentFactor = augmentFactor
        self.labels_dict = {}
        self.fileType = fileType
        self.model_inputs = model_inputs
        self.datasetFolder = datasetFolder
        self.excludeClasses = excludeClasses
        if labelJsonPath is not None:
            self.labels_dict = json.load(open(labelJsonPath, 'r'))
        print("stroke_dataset - {}".format(self.stroke_dataset))

        print('.. Done with attribute settings. Loading the data ...')

        # Loading train, validation and test sets
        if self.load_mode in ['train', 'validation']:
            self.train_set, self.validation_set, self.train_raw, self.validation_raw = self.__load_dataset_splitted(
                stroke_type='SYNTH')
            self.test_set = self.validation_set
            self.test_raw = self.validation_raw
        if self.load_mode == 'test':
            self.test_set, self.test_raw = self.__load_dataset(stroke_type='HUMAN')
            self.train_set, self.validation_set = self.test_set, self.test_set
            self.train_raw, self.validation_raw = self.test_raw, self.test_raw

        self.raw_dataset = self.train_raw, self.validation_raw, self.test_raw
        self.raw_labels = self.train_set[1], self.validation_set[1], self.test_set[1]

        print('.. Done with data loading. Setting up classifier attributes ...')

        self.tuple = self.train_set[0].shape[-1]

        print('Done with DataLoader construction!')

    # Function for reading a sample of the 1dollar class from csv
    def __read_csv_1dollar(self, file_name):
        df = pd.read_csv(file_name, sep=',')
        df = df[['x', 'y']]  # / 100

        # normalize w.r.t to initial point.
        df['x'] -= df['x'].iloc[0]
        df['y'] -= df['y'].iloc[0]

        if (df.isnull().values.any()):
            print(file_name)
            print(df)

        return df.values.tolist()

    # Function designed to load the $1 dataset
    def __load_dataset_on_folder(self, folder_name):

        tensor_x = []
        tensor_y = []
        labels_dict = self.labels_dict
        index = len(self.labels_dict)

        files = os.listdir(folder_name)
        if self.load_mode == 'reduced':
            files = files[::30]

        i = 1
        for file in files:
            if file[0] == '.' or any([ele == file.lower().split('-')[0] for ele in self.excludeClasses]):
                continue

            if i % 100 == 0:
                print("{}%- Loading on {}".format('{0:.2f}'.format(i / len(files) * 100), folder_name))
            i += 1

            file_path = folder_name + '/' + file

            current_label = ''
            if self.stroke_dataset == "Napkin":
                label = file.split('-')[0].lower()
                if label == 'arrow':
                    current_label = 'arrow'
                else:
                    current_label = "non-arrow"
            else:
                current_label = file.split('-')[1]

            if self.stroke_dataset in ['1$', 'Napkin']:
                x = self.__read_csv_1dollar(file_path)
                if len(x) > 75:
                    continue
                tensor_x.append(x)

            if current_label not in labels_dict:
                labels_dict[current_label] = index
                index += 1

            tensor_y.append(labels_dict[current_label])

        print("Loading on {} completed.".format(folder_name))
        print(labels_dict)
        self.labels_dict = labels_dict

        return tensor_x, tensor_y

    def __load_json_data(self):
        subFolders = [f.path for f in os.scandir(self.datasetFolder) if f.is_dir()]
        tensor_x = []
        tensor_y = []
        for folder in subFolders:
            label = folder.split('/')[-1]
            files = [f.path for f in os.scandir(folder) if (not f.is_dir()) and f.path.split('/')[-1][0] != '.']
            for file in files:
                tensor_x.append(readJsonContentTestData(file))
                tensor_y.append(self.labels_dict[label])
        return tensor_x, tensor_y

    # Function designed to load the $1 dataset
    def __load_dataset(self, stroke_type='', preprocess=True):

        # Loading the dataset
        if self.fileType == 'json' and self.load_mode == 'test':
            x, y = self.__load_json_data()
        else:
            x, y = self.__load_dataset_on_folder(self.datasetFolder)
        x, y = np.array(x), np.array(y)

        x_raw = x

        if preprocess:
            x, y, x_raw = self.__preprocess_data(x, y, load_mode='test')

        return (x, y), x_raw

    # Function designed to load the $1 dataset splitted in test and train sets
    def __load_dataset_splitted(self, stroke_type):

        (x, y), x_raw = self.__load_dataset(stroke_type=stroke_type, preprocess=False)

        # Splitting into test and training set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=4, stratify=y)

        # Normalize x_train
        print('preprocessing train set ...')
        x_train, y_train, x_train_raw = self.__preprocess_data(x_train, y_train, augment_data=True, load_mode='train')

        # Normalize x_test
        print('preprocessing test set ...')
        x_test, y_test, x_test_raw = self.__preprocess_data(x_test, y_test, augment_data=False, load_mode='validation')

        return (x_train, y_train), (x_test, y_test), x_train_raw, x_test_raw

    def get_index_to_label(self):
        indexToLabel = {}
        for k, v in self.labels_dict.items():
            indexToLabel[v] = k
        return indexToLabel

    def __preprocess_data(self, x, y, augment_data=False, load_mode=''):

        if augment_data:
            x_augment, y_augment = [], []
            indexToLabel = self.get_index_to_label()
            dataAugmentation = DataAugmentation(augmentFactor=self.augmentFactor)
            for i in range(x.shape[0]):
                x_transformed = dataAugmentation.apply_data_transform(np.array(x[i]), className=indexToLabel[y[i]])
                x_augment.extend(x_transformed)
                y_augment.extend([y[i] for idx in range(self.augmentFactor)])
            x = np.concatenate((x, np.array(x_augment)), axis=0)
            y = np.concatenate((y, np.array(y_augment)), axis=0)

        x = [self.length_normalize_unit_vector(x[i], dims=self.model_inputs) for i in range(x.shape[0])]

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

        return x, y, x_raw

    def length_normalize_unit_vector(self, points, dims='coord_and_tang'):
        # Always compute all the dims and trim at the end
        if len(points) <= 1:
            return points

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
    dl = DataLoader(dataset='Napkin', load_mode='train', labelJsonPath=None,
                    datasetFolder='./dataset/NapkinData/TrainSet/csv',
                    fileType='csv', include_fingerup=False, model_inputs='coord_and_tang',
                    augmentFactor=3, excludeClasses=['line'])

    print(dl.validation_set[0].shape)
    print(dl.train_set[0].shape)
    for k, v in dl.labels_dict.items():
        print('{} - {}'.format(v, k))
