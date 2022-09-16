from numpy.random import seed
from tensorflow.python.ops.control_flow_ops import Switch, switch

seed(0)
from tensorflow import random

random.set_seed(0)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Masking, Input, concatenate, LeakyReLU, Bidirectional
import matplotlib.pyplot as plt
import json


class GestuReNN_GRU:

    def __init__(self, n_labels=11, plot=True, include_fingerup=False, batch_size=128,
                 model_path=None, model_inputs='coord_and_tang', useSaliency=False):
        self.model_with_state = None
        self.arrowModel = None
        self.plot = plot
        self.model_inputs = model_inputs
        self.useSaliency = useSaliency

        # Hyper parameters for optimizing
        self.n_labels = n_labels
        print("----#classes = {}------".format(self.n_labels))
        self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),
                        tf.keras.metrics.MeanAbsoluteError()]  # ['accuracy']
        self.saliency_loss_clf = 'sparse_categorical_crossentropy'
        self.loss_clf_arrow = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.loss_arrowShaftHeadSplit = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_clf = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_reg = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.batch_size = batch_size
        self.epochs = 1000
        # self.opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5, beta_1=0.8, beta_2=0.85)
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6, beta_1=0.9, beta_2=0.95)

        # Setting up checkpoint root
        root = "checkpoints/models/mts"

        # Checkpoints
        self.model_path = root if model_path is None else model_path  # + "/mdcp_robust.ckpt"

        # Loss figure settings
        self.loss_model_path = root if model_path is None else model_path + "/loss_joined_robust.png"

        # model parameters
        self.rnn1_hid_dim = 64
        self.rnn2_hid_dim = 32
        self.arrow_hid_dim = 16

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
        rnn1 = GRU(self.rnn1_hid_dim, input_shape=(None, self.tup), return_sequences=True, reset_after=False,
                   activation=LeakyReLU(alpha=0.1),
                   name='Gate1')(mask)
        drop1 = Dropout(0.1, name='Reg1', seed=0)(rnn1)

        output0 = None
        if self.useSaliency:
            output0 = Dense(2, activation='softmax', name='Saliency_Clf')(drop1)

        rnn_arrow_clf = Bidirectional(GRU(self.arrow_hid_dim, return_sequences=False, reset_after=False,
                                          name='Gate_Clf_Arrow'), merge_mode='concat')(drop1)
        output_arrow = Dense(2, activation='softmax', name='Clf_Arrow')(rnn_arrow_clf)

        rnn_clf = GRU(self.rnn2_hid_dim, return_sequences=True, reset_after=False, activation=LeakyReLU(alpha=0.1),
                      name='Gate_Clf')(drop1)
        # drop_clf = Dropout(0.2, name='Drop_Clf', seed=0)(lstm_clf)
        output_shape = Dense(self.n_labels, activation='softmax', name='Clf')(rnn_clf)

        rnn_reg = GRU(self.rnn2_hid_dim, return_sequences=True, reset_after=False, activation=LeakyReLU(alpha=0.1),
                      name='Gate_Reg')(drop1)
        # drop_reg = Dropout(0.2, name='Drop_Reg', seed=0)(lstm_reg)
        output_reg = Dense(1, activation='sigmoid', name='Reg')(rnn_reg)

        if self.useSaliency:
            self.model = Model(inputs=[visible],
                               outputs=[output0, output_arrow, output_shape, output_reg])
            self.model.compile(loss=[self.saliency_loss_clf, self.loss_clf_arrow, self.custom_loss_clf,
                                     self.custom_loss_reg],
                               optimizer=self.opt, metrics=None)
        else:
            self.model = Model(inputs=[visible], outputs=[output_arrow, output_shape, output_reg])
            self.model.compile(loss=[self.loss_clf_arrow, self.custom_loss_clf, self.custom_loss_reg],
                               optimizer=self.opt,
                               metrics={'Clf_Arrow': self.custom_metric_arrow, 'Clf': self.custom_metric_shape})

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

    def custom_loss_arrowShaftHeadSplit(self, y_gt, y_pred):
        loss = self.loss_arrowShaftHeadSplit(y_gt, y_pred) * 100
        return loss

    def custom_metric_shape(self, y_true, y_pred):
        y = tf.cast(y_true[:, 0, 0], tf.int64)  # shape(batch_size,)
        lastIndex = tf.argmax(y_true[:, :, 1], axis=1)  # (batch)
        pred = tf.argmax(y_pred, axis=2)  # shape(batch,seq_len)
        pred = tf.gather(pred, lastIndex, axis=1, batch_dims=1)  # (batch,)
        is_equal = tf.math.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
        return accuracy

    def custom_metric_arrow(self, y_true, y_pred):
        y = tf.cast(y_true[:, 0], tf.int64)  # shape(batch_size,)
        pred = tf.argmax(y_pred, axis=1)  # shape(batch,)
        is_equal = tf.math.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
        return accuracy

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

    def fit_model(self, train_clf, test_clf, train_reg, test_reg, y_arrow_train, y_arrow_test, y_saliency_train=None,
                  y_saliency_test=None, wandbCallBacks=None):

        (x_train, y_train_clf), (x_test, y_test_clf) = train_clf, test_clf
        (_, y_train_reg), (_, y_test_reg) = train_reg, test_reg

        y_arrowBinary_train, y_arrowSplitMask_train = y_arrow_train
        y_arrowBinary_test, y_arrowSplitMask_test = y_arrow_test

        # Setting up the checkpoint
        cp_callback = []
        if wandbCallBacks is None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor="val_Clf_custom_metric_shape",
                                                             mode="max",
                                                             verbose=1)
        else:
            cp_callback = wandbCallBacks

        y_train_clf, y_train_reg = self.concatenateWeight(y_train_clf, y_train_reg)
        y_test_clf, y_test_reg = self.concatenateWeight(y_test_clf, y_test_reg)
        print("y_train_clf - {} \n y_test_clf - {} \n y_train_reg - {}".format(y_train_clf.shape, y_test_clf.shape,
                                                                               y_train_reg.shape))
        # Training the net
        history = None
        if y_saliency_test is None or y_saliency_train is None:
            history = self.model.fit(x_train,
                                     {"Clf_Arrow": y_arrowBinary_train,
                                      "Clf": y_train_clf,
                                      "Reg": y_train_reg},
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(
                                         x_test, {"Clf_Arrow": y_arrowBinary_test,
                                                  "Clf": y_test_clf,
                                                  "Reg": y_test_reg}),
                                     callbacks=[cp_callback])
        else:
            history = self.model.fit(x_train, {"Clf_Arrow": y_arrowBinary_train,
                                               "Clf": y_train_clf,
                                               "Reg": y_train_reg,
                                               "Saliency_Clf": y_saliency_train},
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(x_test,
                                                      {"Clf_Arrow": y_arrowBinary_test,
                                                       "Clf": y_test_clf, "Reg": y_test_reg,
                                                       "Saliency_Clf": y_saliency_test}),
                                     callbacks=[cp_callback])
        # Plotting the losses
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.savefig(self.loss_model_path)
        # if self.plot:
        #     plt.show()
        # plt.clf()

    def load_model(self, by_name=False):
        print(self.model_path)
        self.model.load_weights(self.model_path, by_name=by_name)

    def define_model_with_state(self):
        # model inputs
        curve_input = tf.keras.Input(shape=(None, self.tup))
        rnn1_state_input = tf.keras.Input(shape=self.rnn1_hid_dim)
        rnn_clf_state_input = tf.keras.Input(shape=self.rnn2_hid_dim)
        rnn_reg_state_input = tf.keras.Input(shape=self.rnn2_hid_dim)

        # RNN Definition
        rnn1 = GRU(self.rnn1_hid_dim, input_shape=(None, self.tup), return_sequences=True, stateful=False,
                   activation=LeakyReLU(alpha=0.1), return_state=True, reset_after=False, name='Gate1')
        rnn_clf = GRU(self.rnn2_hid_dim, return_sequences=False, activation=LeakyReLU(alpha=0.1),
                      stateful=False, return_state=True, reset_after=False, name='Gate_Clf')
        rnn_reg = GRU(self.rnn2_hid_dim, return_sequences=False, activation=LeakyReLU(alpha=0.1),
                      stateful=False, return_state=True, reset_after=False, name='Gate_Reg')

        # model architecture
        rnn1_hid_output, rnn1_state_output_h = rnn1(curve_input, initial_state=rnn1_state_input)
        drop1 = Dropout(0.1, name='Reg1', seed=0)(rnn1_hid_output)

        rnn_clf_hid_output, rnn_clf_state_output_h = rnn_clf(drop1, initial_state=rnn_clf_state_input)
        output1 = Dense(self.n_labels, activation='softmax', name='Clf')(rnn_clf_hid_output)

        rnn_reg_hid_output, rnn_reg_state_output_h = rnn_reg(drop1, initial_state=rnn_reg_state_input)
        output2 = Dense(1, activation='sigmoid', name='Reg')(rnn_reg_hid_output)

        model_with_state = Model(inputs=[rnn1_state_input, rnn_clf_state_input, rnn_reg_state_input, curve_input],
                                 outputs=[rnn1_state_output_h, rnn_clf_state_output_h, rnn_reg_state_output_h, output1,
                                          output2])
        return model_with_state

    def arrowModelDefinition(self):
        # model inputs
        curve_input = tf.keras.Input(shape=(None, self.tup))

        # RNN Definition
        rnn1 = GRU(self.rnn1_hid_dim, input_shape=(None, self.tup), return_sequences=True, stateful=False,
                   activation=LeakyReLU(alpha=0.1), return_state=False, reset_after=False, name='Gate1')
        rnn_arrow_clf = Bidirectional(GRU(self.arrow_hid_dim, return_sequences=False, reset_after=False,
                                          name='Gate_Clf_Arrow'), merge_mode='concat')
        # model architecture
        rnn1_hid_output = rnn1(curve_input)
        drop1 = Dropout(0.1, name='Reg1', seed=0)(rnn1_hid_output)

        rnn_arrow_clf_hid_output = rnn_arrow_clf(drop1)
        output_arrow = Dense(2, activation='softmax', name='Clf_Arrow')(rnn_arrow_clf_hid_output)

        arrowModel = Model(inputs=[curve_input], outputs=[output_arrow])
        return arrowModel

    def load_model_with_state(self, by_name=False):
        self.load_model(by_name=by_name)
        self.model.save_weights("/tmp/model_mts.h5")
        self.model_with_state = self.define_model_with_state()
        self.model_with_state.load_weights("/tmp/model_mts.h5", by_name=True)
        self.arrowModel = self.arrowModelDefinition()
        self.arrowModel.load_weights("/tmp/model_mts.h5", by_name=True)


class BinaryArrowModel:

    def __init__(self, n_labels=2, plot=True, include_fingerup=False, batch_size=128,
                 model_path=None, model_inputs='coord_and_tang'):
        self.arrowModel = None
        self.plot = plot
        self.model_inputs = model_inputs

        # Hyper parameters for optimizing
        self.n_labels = n_labels
        print("----#classes = {}------".format(self.n_labels))
        self.loss_clf_arrow = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.arrowPrecision = tf.keras.metrics.Precision(thresholds=0.5, class_id=1, name="arrowPrecisioin")
        self.arrowRecall = tf.keras.metrics.Recall(thresholds=0.5, class_id=1, name='arrowRecall')
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.batch_size = batch_size
        self.epochs = 1000
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5, beta_1=0.9, beta_2=0.95)

        # Setting up checkpoint root
        root = "checkpoints/models/mts"

        # Checkpoints
        self.model_path = root if model_path is None else model_path  # + "/mdcp_robust.ckpt"

        # Loss figure settings
        self.loss_model_path = root if model_path is None else model_path + "/loss_joined_robust.png"

        # model parameters
        self.arrow_hid_dim = 16

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

        rnn_arrow_clf = Bidirectional(GRU(self.arrow_hid_dim, return_sequences=False, reset_after=False,
                                          name='Gate_Clf_Arrow'), merge_mode='concat')(mask)
        output_arrow = Dense(2, activation='softmax', name='Clf_Arrow')(rnn_arrow_clf)

        self.model = Model(inputs=[visible], outputs=[output_arrow])
        self.model.compile(loss=[self.loss_clf_arrow], optimizer=self.opt,
                           metrics=[self.accuracy])

    def fit_model(self, train_clf, test_clf, y_arrow_train, y_arrow_test, wandbCallBacks=None):

        (x_train, y_train_clf), (x_test, y_test_clf) = train_clf, test_clf

        y_arrowBinary_train, y_arrowSplitMask_train = y_arrow_train
        y_arrowBinary_test, y_arrowSplitMask_test = y_arrow_test

        # Setting up the checkpoint
        cp_callback = []
        if wandbCallBacks is None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor="val_loss",
                                                             mode="min",
                                                             verbose=1)
        else:
            cp_callback = wandbCallBacks

        # Training the net
        history = self.model.fit(x_train,
                                 {"Clf_Arrow": y_arrowBinary_train},
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(
                                     x_test, {"Clf_Arrow": y_arrowBinary_test}),
                                 callbacks=[cp_callback])

        # Plotting the losses
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.savefig(self.loss_model_path)
        # if self.plot:
        #     plt.show()
        # plt.clf()

    def load_model(self, by_name=False):
        print(self.model_path)
        self.model.load_weights(self.model_path, by_name=by_name)

    def arrowModelDefinition(self):
        # model inputs
        curve_input = tf.keras.Input(shape=(None, self.tup))

        # RNN Definition
        rnn_arrow_clf = Bidirectional(GRU(self.arrow_hid_dim, return_sequences=False, reset_after=False,
                                          name='Gate_Clf_Arrow'), merge_mode='concat')
        # model architecture
        rnn_arrow_clf_hid_output = rnn_arrow_clf(curve_input)
        output_arrow = Dense(2, activation='softmax', name='Clf_Arrow')(rnn_arrow_clf_hid_output)

        arrowModel = Model(inputs=[curve_input], outputs=[output_arrow])
        return arrowModel

    def load_model_with_state(self, by_name=False):
        self.load_model(by_name=by_name)
        self.model.save_weights("/tmp/model_mts.h5")
        self.arrowModel = self.arrowModelDefinition()
        self.arrowModel.load_weights("/tmp/model_mts.h5", by_name=True)

if __name__ == "__main__":
    params = {
        "project_name": "Arrow recognition - bidirectional rnn for arrow clf",
        "pad": True,
        "include_fingerup": False,
        "model_inputs": 'coord_and_tang',
        "test_size": 0.2,
        "load_mode": 'train',
        "augmentFactor": 6,
        "dataFile": '/content/drive/MyDrive/second_layer_ai_model/SketchAI/SketchAI/dataset/NapkinData/train.json',
        "batchSize": 64,
        "useSaliency": False,
        "isResample": False,
        "modelPath": '/content/drive/MyDrive/second_layer_ai_model/SketchAI/SketchAI/checkpoints/models/BinaryArrowModel'
    }
    model = BinaryArrowModel(plot=False, model_inputs=params["model_inputs"], batch_size=params["batchSize"],
                             model_path=params["modelPath"], include_fingerup=params["include_fingerup"])
