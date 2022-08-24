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
    def __init__(self, pad=True, include_fingerup=False, use_tangents=False, test_size=0.2,
                 method='G3', dataset='1$', load_mode='full', augmentFactor=3,
                 datasetFolder=Napkin_HUMAN, fileType='', labelJsonPath=None, excludeClasses=['line']):

        print('Starting the DataLoader construction ...')

        # Setting general data loader attributes
        self.use_padding = pad
        self.include_fingerup = include_fingerup
        self.use_tangents = use_tangents
        self.test_size = test_size
        self.method = method
        self.stroke_dataset = dataset
        self.load_mode = load_mode
        self.augmentFactor = augmentFactor
        self.labels_dict = {}
        self.fileType = fileType
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

    # Function for reading a sample of the 1dollar class from csv
    def __read_csv_1dollar(self, file_name):
        df = pd.read_csv(file_name, sep=',')
        df = df[['x', 'y']]  # / 100

        # normalize w.r.t to initial point.
        df['x'] -= df['x'].iloc[0]
        df['y'] -= df['y'].iloc[0]

        # Adding fingerup serie
        # if self.include_fingerup:
        #     df['finger_up'] = 0
        #     # df['finger_up'].iloc[-1] = 1
        #     df.loc[len(df) - 1, 'finger_up'] = 1

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
            if self.method == 'G3':
                if self.stroke_dataset == "Napkin":
                    label = file.split('-')[0].lower()
                    if label == 'square':
                        current_label = 'rectangle'
                    elif 'curly' in label:
                        current_label = 'curly_braces'
                    elif 'bracket' in label:
                        current_label = 'bracket'
                    else:
                        current_label = label

                else:
                    current_label = file.split('-')[1]

            if current_label not in labels_dict:
                labels_dict[current_label] = index
                index += 1

            if self.method == 'G3':
                if self.stroke_dataset in ['1$', 'Napkin']:
                    tensor_x.append(self.__read_csv_1dollar(file_path))

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

        if self.use_tangents:
            x = [self.length_normalize_unit_vector(x[i], dims='coord_and_tang') for i in range(x.shape[0])]

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

    def __get_labels_repeated(self):
        x_train, y_train = self.train_set
        x_test, y_test = self.test_set
        x_validation, y_validation = self.validation_set

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
            ys[i] = np.pad(np.arange(1,len(xs[i])+1), (0, pad_gap), 'constant', constant_values=(0, 0))
            ys[i] = np.array(ys[i] / len(xs[i]), dtype='float32')

        return ys

    def length_normalize_unit_vector(self, points, dims='coord_and_tang'):
        # Always compute all the dims and trim at the end
        if len(points) <= 1:
            return points

        np_points = np.array(points, dtype=np.float32)
        output = [[np_points[0][0], np_points[0][1], 0.0, 0.0]]
        for index in range(1, np_points.shape[0]):
            pt1 = np_points[index]
            pt1_prev = np_points[index - 1]
            if np.linalg.norm(pt1 - pt1_prev) == 0:
                print("two same consecutive points are found, skipping one in the output seq.")
                continue
            unit = (pt1 - pt1_prev) / np.linalg.norm(pt1 - pt1_prev)
            output.append([pt1[0], pt1[1], unit[0], unit[1]])

        np_output = np.array(output, dtype=np.float32)

        if dims == "coord_and_tang":
            return np_output
        if dims == "coordinates":
            return np_output[:, 0:2]
        if dims == "tangents":
            return np_output[:, 2:]

        raise RuntimeError(f"Unexpected dims value ${dims}")


if __name__ == "__main__":
    arr = np.array(
        [[1.25836086e-03, -1.49348695e-03], [-7.04729025e-01, 1.87958431e+01], [-2.61200568e+00, 3.47530422e+01],
         [-5.49237353e+00, 4.59087434e+01], [-8.99364300e+00, 5.08975409e+01], [-1.51748133e+01, 5.13679206e+01],
         [-2.89841870e+01, 5.18351976e+01], [-2.96602450e+01, 5.18581375e+01], [-3.03141072e+01, 5.18955519e+01],
         [-3.14238686e+01, 5.19760318e+01], [-3.27074356e+01, 5.20762529e+01], [-3.71374664e+01, 5.24270242e+01],
         [-3.96655248e+01, 5.26104393e+01], [-4.44606242e+01, 5.28120252e+01], [-4.87777089e+01, 5.28175091e+01],
         [-5.25098800e+01, 5.26233811e+01], [-5.55734810e+01, 5.22402757e+01], [-5.72252867e+01, 5.17052993e+01],
         [-5.88387119e+01, 5.06984808e+01], [-6.03347840e+01, 4.92688479e+01], [-6.16407924e+01, 4.74842857e+01],
         [-6.83684676e+01, 3.31268802e+01], [-7.15562068e+01, 1.83249026e+01], [-7.06364475e+01, 5.71426806e+00],
         [-6.57641282e+01, -2.44940946e+00], [-5.78280846e+01, -4.71015396e+00], [-5.05661507e+01, -3.97581238e+00],
         [-3.30276810e+01, -2.19499372e+00], [-3.30272556e+01, -2.19495047e+00], [-3.30268301e+01, -2.19490722e+00],
         [-3.30268301e+01, -2.19490722e+00], [-3.30268301e+01, -2.19490722e+00], [-1.54861693e+01, -4.13064888e-01],
         [-8.22462483e+00, 3.22006241e-01], [-8.39172765e+00, 7.33085226e+00], [-8.54978175e+00, 1.43456467e+01],
         [-8.71538775e+00, 2.13614662e+01], [-8.87514694e+00, 2.83779021e+01], [-9.04106645e+00, 3.53878464e+01],
         [-9.20199655e+00, 4.23978494e+01], [-9.36588722e+00, 4.94112974e+01], [-9.52887675e+00, 5.64200785e+01],
         [-1.56929907e+01, 5.66691571e+01], [-2.18594034e+01, 5.69171653e+01], [-2.80244213e+01, 5.71612903e+01],
         [-3.41833166e+01, 5.73990640e+01], [-4.03492500e+01, 5.76420692e+01], [-4.65238765e+01, 5.78946556e+01],
         [-5.26883003e+01, 5.81353305e+01], [-5.88525219e+01, 5.83795118e+01]])
    print(arr.shape)
    out = length_normalize_unit_vector_debug(arr)
    print(out)
    print(out.shape)
    print(np.isnan(out).any())
