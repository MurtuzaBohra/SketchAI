from numpy.random import seed

seed(0)
from tensorflow import random

random.set_seed(0)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Masking, Input, concatenate
import matplotlib.pyplot as plt
import json

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class GestuReNN:

    def __init__(self, dataset='1$', topology='stl', labelJsonPath='', plot=True, include_fingerup=False,
                 robust_normalization=True, batch_size=128, model_path=None):
        self.plot = plot
        self.topology = topology
        self.robust_normalization = robust_normalization

        if labelJsonPath is not None:
            self.gesture_dict_1dollar = json.load(open(labelJsonPath, 'r'))
        else:
            self.gesture_dict_1dollar = {
                0: 'arrow', 1: 'caret', 2: 'check', 3: 'O',
                4: 'delete', 5: '{', 6: '[', 7: 'pig-tail',
                8: '?', 9: 'rectangle', 10: '}', 11: ']',
                12: 'star', 13: 'triangle', 14: 'V', 15: 'X'
            }

        self.gesture_dict_ndollar = {
            0: 'arrowhead', 1: '*', 2: 'D', 3: '!',
            4: 'five-point-star', 5: 'H', 6: 'half-note', 7: 'I',
            8: 'line', 9: 'N', 10: 'null', 11: 'P',
            12: 'pitchfork', 13: 'six-point-star', 14: 'T', 15: 'X',
        }

        self.stroke_mapping_ndollar = np.array([2, 3, 2, 2, 1, 3, 2, 3, 1, 3, 2, 2, 2, 2, 2, 2])

        # Hyperparameters for optimizing
        self.n_labels = len(self.gesture_dict_1dollar)
        print("----#classes = {}------".format(self.n_labels))
        self.metrics = ['accuracy']
        self.loss_clf = 'sparse_categorical_crossentropy'
        self.loss_reg = 'mse'
        self.batch_size = batch_size
        self.epochs = 200
        # self.opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5, beta_1=0.8, beta_2=0.85)
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4, beta_1=0.8, beta_2=0.85)

        # Setting up checkpoint root
        root = "checkpoints/models/" + dataset

        # Checkpoints
        self.sts_clf_path, self.sts_reg_path = root + "/cp_sml_robust.ckpt", root + "/rgcp_sml_robust.ckpt"
        self.stl_clf_path, self.stl_reg_path = root + "/cp_robust.ckpt", root + "/rgcp_robust.ckpt"
        self.mts_path = root if model_path is None else model_path + "/mdcp_robust.ckpt"
        self.mtm_path = root + "/mdvarcp_robust.ckpt"

        # Loss figure settings
        self.sts_clf_loss, self.sts_reg_loss = root + "/loss_sml_clf_robust.png", root + "/loss_sml_reg_robust.png"
        self.stl_clf_loss, self.stl_reg_loss = root + "/loss_clf_robust.png", root + "/loss_reg_robust.png"
        self.mts_loss = root + "/loss_joined_robust.png"
        self.mtm_loss = root + "/loss_joined_var_robust.png"

        if self.topology == 'sts':
            self.classifier_path, self.loss_clf_path = self.sts_clf_path, self.sts_clf_loss
            self.regressor_path, self.loss_reg_path = self.sts_reg_path, self.sts_reg_loss
        if self.topology == 'stl':
            self.classifier_path, self.loss_clf_path = self.stl_clf_path, self.stl_clf_loss
            self.regressor_path, self.loss_reg_path = self.stl_reg_path, self.stl_reg_loss
        if self.topology == 'mts':
            self.model_path = self.mts_path
            self.loss_model_path = self.mts_loss
        if self.topology == 'mtm':
            self.model_path = self.mtm_path
            self.loss_model_path = self.mtm_loss

        self.tup = 2
        if dataset == 'N$':
            self.tup = 3
        if include_fingerup:
            self.tup += 1

        # Setting up classifier and regressor
        self.classifier = Sequential()
        self.regressor = Sequential()

        # Masking layer
        self.classifier.add(Masking(mask_value=0.))
        self.regressor.add(Masking(mask_value=0., input_shape=(None, self.tup)))

        # Adding gates and (eventually) dropout
        self.classifier.add(LSTM(256, input_shape=(None, self.tup), return_sequences=True))
        if topology != 'sts':
            self.classifier.add(Dropout(0.2, seed=0))
            self.classifier.add(LSTM(64, input_shape=(None, self.tup), return_sequences=True))
            self.classifier.add(Dropout(0.2, seed=0))
        self.regressor.add(LSTM(256, input_shape=(None, self.tup), return_sequences=True))
        if topology != 'sts':
            self.regressor.add(Dropout(0.2, seed=0))
            self.regressor.add(LSTM(64, input_shape=(None, self.tup), return_sequences=True))
            self.regressor.add(Dropout(0.2, seed=0))

        # Final Classifier and Regressor layers
        self.classifier.add(Dense(self.n_labels, activation='softmax'))  # Probability distribution of 16 classes
        self.regressor.add(Dense(1, activation='sigmoid'))  # Probability distribution of completion 0-1

        # Compiling the model
        self.classifier.compile(loss=self.loss_clf, optimizer=self.opt, metrics=self.metrics)
        self.regressor.compile(loss=self.loss_reg, optimizer=self.opt, metrics=self.metrics)

        # Joined model
        visible = Input(shape=(None, self.tup), name='Input')
        mask = Masking(mask_value=0, name='Masking')(visible)
        lstm1 = LSTM(256, input_shape=(None, self.tup), return_sequences=True, name='Gate1')(mask)
        drop1 = Dropout(0.2, name='Reg1', seed=0)(lstm1)

        if self.topology == 'mtm':
            lstm2 = LSTM(256, input_shape=(None, self.tup), return_sequences=True, name='Gate2')(mask)
            drop2 = Dropout(0.2, name='Reg2', seed=0)(lstm2)

        lstm_clf = LSTM(64, input_shape=(None, self.tup), return_sequences=True, name='Gate_Clf')(drop1)
        drop_clf = Dropout(0.2, name='Drop_Clf', seed=0)(lstm_clf)
        output1 = Dense(self.n_labels, activation='softmax', name='Clf')(drop_clf)

        lstm_reg = LSTM(64, input_shape=(None, self.tup), return_sequences=True, name='Gate_Reg')(drop1)
        drop_reg = Dropout(0.2, name='Drop_Reg', seed=0)(lstm_reg)
        output2 = Dense(1, activation='sigmoid', name='Reg')(drop_reg)

        if self.topology == 'mtm':
            lstm_reg = LSTM(64, input_shape=(None, self.tup), return_sequences=True, name='Gate_Reg')(drop2)
            drop_reg = Dropout(0.2, name='Drop_Reg', seed=0)(lstm_reg)

            concat1 = concatenate([drop_clf, drop_reg])
            concat2 = concatenate([drop_clf, drop_reg])

            output1 = Dense(self.n_labels, activation='softmax', name='Clf')(concat1)
            output2 = Dense(1, activation='sigmoid', name='Reg')(concat2)

        # model = Model(inputs=visible, outputs=output)
        self.model = Model(inputs=[visible], outputs=[output1, output2])
        self.model.compile(loss=[self.loss_clf, self.loss_reg], optimizer=self.opt, metrics=self.metrics)

    def fit_model(self, train_clf, test_clf, train_reg, test_reg):

        (x_train, y_train_clf), (x_test, y_test_clf) = train_clf, test_clf
        (_, y_train_reg), (_, y_test_reg) = train_reg, test_reg

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Training the net
        history = self.model.fit(x_train, {"Clf": y_train_clf, "Reg": y_train_reg},
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(x_test, {"Clf": y_test_clf, "Reg": y_test_reg}),
                                 callbacks=[cp_callback])

        # Plotting the losses
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(self.loss_model_path)
        if self.plot:
            plt.show()
        plt.clf()

    def fit_classifier(self, x_train, y_train, x_test, y_test):

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.classifier_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Training the net
        history = self.classifier.fit(x_train, y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      validation_data=(x_test, y_test),
                                      callbacks=[cp_callback])

        # Plotting the losses
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(self.loss_clf_path)
        if self.plot:
            plt.show()
        plt.clf()

    def fit_clf(self, train, test):
        self.fit_classifier(train[0], train[1], test[0], test[1])

    def fit_regressor(self, x_train, y_train, x_test, y_test):

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.regressor_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = self.regressor.fit(x_train, y_train,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(x_test, y_test),
                                     callbacks=[cp_callback])

        # Plotting the losses
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(self.loss_reg_path)
        if self.plot:
            plt.show()
        plt.clf()

    def fit_reg(self, train, test):
        self.fit_regressor(train[0], train[1], test[0], test[1])

    def load_clf(self):
        self.classifier.load_weights(self.classifier_path)

    def load_reg(self):
        self.regressor.load_weights(self.regressor_path)

    def load_model(self):
        if self.topology == 'mts' or self.topology == 'mtm':
            print(self.model_path)
            self.model.load_weights(self.model_path)
        else:
            self.load_reg()
            self.load_clf()

    def classify(self, gesture):
        prediction = self.classifier.predict([gesture])
        return np.argmax(prediction, axis=1)[0]
