from numpy.random import seed
from tensorflow.python.ops.control_flow_ops import Switch, switch

seed(0)
from tensorflow import random

random.set_seed(0)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Masking, Input, concatenate, LeakyReLU
import matplotlib.pyplot as plt
import json


class GestuReNN_GRU:

    def __init__(self, labelJsonPath='', plot=True, include_fingerup=False, batch_size=128,
                 model_path=None, model_inputs='coord_and_tang', useSaliency=False):
        self.model_with_state = None
        self.plot = plot
        self.model_inputs = model_inputs
        self.useSaliency = useSaliency

        if labelJsonPath is not None:
            self.gesture_dict_1dollar = json.load(open(labelJsonPath, 'r'))
        else:
            self.gesture_dict_1dollar = {
                0: 'arrow', 1: 'caret', 2: 'check', 3: 'O',
                4: 'delete', 5: '{', 6: '[', 7: 'pig-tail',
                8: '?', 9: 'rectangle', 10: '}', 11: ']',
                12: 'star', 13: 'triangle', 14: 'V', 15: 'X'
            }

        # Hyper parameters for optimizing
        self.n_labels = len(self.gesture_dict_1dollar)
        print("----#classes = {}------".format(self.n_labels))
        self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),
                        tf.keras.metrics.MeanAbsoluteError()]  # ['accuracy']
        self.saliency_loss_clf = 'sparse_categorical_crossentropy'
        self.loss_clf_arrow = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.loss_clf = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_reg = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.batch_size = batch_size
        self.epochs = 1000
        # self.opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5, beta_1=0.8, beta_2=0.85)
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4, beta_1=0.8, beta_2=0.85)

        # Setting up checkpoint root
        root = "checkpoints/models/mts"

        # Checkpoints
        self.model_path = root if model_path is None else model_path + "/mdcp_robust.ckpt"

        # Loss figure settings
        self.loss_model_path = root if model_path is None else model_path + "/loss_joined_robust.png"

        # model parameters
        self.lstm1_hid_dim = 64
        self.lstm2_hid_dim = 32

        self.tup = 2
        if self.model_inputs == 'coord_and_tang':
            self.tup += 2
        if self.model_inputs == 'tangents_and_norm':
            self.tup += 1
        if self.model_inputs == 'coord_tang_and_norm':
            self.tup += 3
        if include_fingerup:
            self.tup += 1

        # Joined model
        visible = Input(shape=(None, self.tup), name='Input')
        mask = Masking(mask_value=0, name='Masking')(visible)
        lstm1 = GRU(self.lstm1_hid_dim, input_shape=(None, self.tup), return_sequences=True, reset_after=False,
                    activation=LeakyReLU(alpha=0.1),
                    name='Gate1')(mask)
        drop1 = Dropout(0.1, name='Reg1', seed=0)(lstm1)

        output0 = None
        if self.useSaliency:
            output0 = Dense(2, activation='softmax', name='Saliency_Clf')(drop1)

        lstm_arrow_clf = GRU(self.lstm2_hid_dim, input_shape=(None, self.tup), return_sequences=False,
                             reset_after=False,
                             activation=LeakyReLU(alpha=0.1),
                             name='Gate_Clf_Arrow')(drop1)
        output_arrow = Dense(2, activation='softmax', name='Clf_Arrow')(lstm_arrow_clf)

        lstm_clf = GRU(self.lstm2_hid_dim, input_shape=(None, self.tup), return_sequences=True, reset_after=False,
                       activation=LeakyReLU(alpha=0.1),
                       name='Gate_Clf')(drop1)
        # drop_clf = Dropout(0.2, name='Drop_Clf', seed=0)(lstm_clf)
        output1 = Dense(self.n_labels, activation='softmax', name='Clf')(lstm_clf)

        lstm_reg = GRU(self.lstm2_hid_dim, input_shape=(None, self.tup), return_sequences=True, reset_after=False,
                       activation=LeakyReLU(alpha=0.1),
                       name='Gate_Reg')(drop1)
        # drop_reg = Dropout(0.2, name='Drop_Reg', seed=0)(lstm_reg)
        output2 = Dense(1, activation='sigmoid', name='Reg')(lstm_reg)

        if self.useSaliency:
            self.model = Model(inputs=[visible], outputs=[output0, output_arrow, output1, output2])
            self.model.compile(
                loss=[self.saliency_loss_clf, self.loss_clf_arrow, self.custom_loss_clf, self.custom_loss_reg],
                optimizer=self.opt, metrics=None)
        else:
            self.model = Model(inputs=[visible], outputs=[output_arrow, output1, output2])
            self.model.compile(loss=[self.loss_clf_arrow, self.custom_loss_clf, self.custom_loss_reg],
                               optimizer=self.opt, metrics=None)

    def custom_loss_clf(self, y_clf_gt, y_clf_pred):
        y0_gt = tf.expand_dims(y_clf_gt[:, :, 0], axis=2, name=None)
        loss = self.loss_clf(y0_gt, y_clf_pred) * y_clf_gt[:, :, 1]
        loss = tf.reduce_sum(loss)
        return loss

    def custom_loss_reg(self, y_reg_gt, y_reg_pred):
        # y_gt = tf.squeeze(y_reg_gt, axis=2)
        y0_gt = tf.expand_dims(y_reg_gt[:, :, 0], axis=2, name=None)
        loss = self.loss_reg(y0_gt, y_reg_pred) * y_reg_gt[:, :, 1]
        loss = tf.reduce_sum(loss)
        return loss

    def concatenateWeight(self, y_clf, y_reg):
        weight = y_reg
        y_clf = np.expand_dims(y_clf, axis=2)
        for i in range(y_clf.shape[0]):
            for j in range(y_clf.shape[1]):
                if y_clf[i, 0, 0] == 5:  # in case of arrow, weights are zero.
                    weight[i, j, 0] = 0
        y_clf = np.concatenate((y_clf, weight), axis=2)
        y_reg = np.concatenate((y_reg, weight), axis=2)
        return y_clf, y_reg

    def fit_model(self, train_clf, test_clf, train_reg, test_reg, y_train_binary, y_test_binary, y_saliency_train=None,
                  y_saliency_test=None):

        (x_train, y_train_clf), (x_test, y_test_clf) = train_clf, test_clf
        (_, y_train_reg), (_, y_test_reg) = train_reg, test_reg

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         monitor="val_loss",
                                                         mode="min",
                                                         verbose=1)
        y_train_clf, y_train_reg = self.concatenateWeight(y_train_clf, y_train_reg)
        y_test_clf, y_test_reg = self.concatenateWeight(y_test_clf, y_test_reg)
        print("y_train_clf shape - {} \n y_test_clf shape - {} \n y_train_reg shape - {}".format(y_train_clf.shape,
                                                                                                 y_test_clf.shape,
                                                                                                 y_train_reg.shape))
        # Training the net
        history = None
        if y_saliency_test is None or y_saliency_train is None:
            history = self.model.fit(x_train,
                                     {"Clf_Arrow": y_train_binary, "Clf": y_train_clf, "Reg": y_train_reg},
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(
                                     x_test, {"Clf_Arrow": y_test_binary, "Clf": y_test_clf, "Reg": y_test_reg}),
                                     callbacks=[cp_callback])
        else:
            history = self.model.fit(x_train, {"Clf_Arrow": y_train_binary, "Clf": y_train_clf, "Reg": y_train_reg,
                                               "Saliency_Clf": y_saliency_train},
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(x_test,
                                                      {"Clf_Arrow": y_test_binary, "Clf": y_test_clf, "Reg": y_test_reg,
                                                       "Saliency_Clf": y_saliency_test}),
                                     callbacks=[cp_callback])
        # Plotting the losses
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(self.loss_model_path)
        if self.plot:
            plt.show()
        plt.clf()

    def load_model(self):
        print(self.model_path)
        self.model.load_weights(self.model_path)

    def define_model_with_state(self):
        # model inputs
        curve_input = tf.keras.Input(shape=(None, self.tup))
        lstm1_state_input = tf.keras.Input(shape=self.lstm1_hid_dim)
        lstm_clf_state_input = tf.keras.Input(shape=self.lstm2_hid_dim)
        lstm_reg_state_input = tf.keras.Input(shape=self.lstm2_hid_dim)

        # RNN Definition
        lstm1 = GRU(self.lstm1_hid_dim, input_shape=(None, self.tup), return_sequences=True,
                    activation=LeakyReLU(alpha=0.1),
                    stateful=False, return_state=True, reset_after=False, name='Gate1')
        lstm_clf = GRU(self.lstm2_hid_dim, return_sequences=False, activation=LeakyReLU(alpha=0.1),
                       stateful=False, return_state=True, reset_after=False, name='Gate_Clf')
        lstm_reg = GRU(self.lstm2_hid_dim, return_sequences=False, activation=LeakyReLU(alpha=0.1),
                       stateful=False, return_state=True, reset_after=False, name='Gate_Reg')

        # model architecture
        lstm1_hid_output, lstm1_state_output_h = lstm1(curve_input, initial_state=lstm1_state_input)
        drop1 = Dropout(0.1, name='Reg1', seed=0)(lstm1_hid_output)

        lstm_clf_hid_output, lstm_clf_state_output_h = lstm_clf(drop1, initial_state=lstm_clf_state_input)
        # drop_clf = Dropout(0.2, name='Drop_Clf', seed=0)(lstm_clf_hid_output)
        output1 = Dense(self.n_labels, activation='softmax', name='Clf')(lstm_clf_hid_output)

        lstm_reg_hid_output, lstm_reg_state_output_h = lstm_reg(drop1, initial_state=lstm_reg_state_input)
        # drop_reg = Dropout(0.2, name='Drop_Reg', seed=0)(lstm_reg_hid_output)
        output2 = Dense(1, activation='sigmoid', name='Reg')(lstm_reg_hid_output)

        model_with_state = Model(inputs=[lstm1_state_input, lstm_clf_state_input, lstm_reg_state_input, curve_input],
                                 outputs=[lstm1_state_output_h, lstm_clf_state_output_h, lstm_reg_state_output_h,
                                          output1, output2])
        return model_with_state

    def load_model_with_state(self):
        self.load_model()
        self.model.save_weights("/tmp/model_mts.h5")
        self.model_with_state = self.define_model_with_state()
        self.model_with_state.load_weights("/tmp/model_mts.h5", by_name=True)
