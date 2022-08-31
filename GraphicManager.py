import time
import numpy as np
import matplotlib.pyplot as plt
from GestuReNN_mts import GestuReNN, GestuReNN_GRU, GestuReNN_mts_without_regression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({'font.size': 14})
stroke_mapping_ndollar = np.array([2, 3, 2, 2, 1, 3, 2, 3, 1, 3, 2, 2, 2, 2, 2, 2])


class GraphicManager:

    def __init__(self, dataset, n_bins=100, acceptance_window=0.1, margin=0.1, save=False, plot=True):
        self.n_bins = n_bins
        self.acceptance_window = acceptance_window
        self.save = save
        self.plot = plot
        self.iterations = 65  # A number of iterations in test phase is needed for gpu memory issues
        self.stroke_dataset = dataset
        self.margin = margin

    def plot_examples(self, model, x, y_reg, n_examples=100):

        # Predicting the values
        clf_pred, reg_pred, _ = self.__make_predictions(model, x)

        mask = x.sum(-1) != 0

        for i in range(n_examples):

            # Setting up the figure
            fig = plt.figure(figsize=(12, 5))
            ax_clf = fig.add_subplot(1, 2, 1)
            ax_reg = fig.add_subplot(1, 2, 2)

            # Setting axes limits
            ax_clf.set_xlim((0 - self.margin, 1 + self.margin))
            ax_clf.set_ylim((1 + self.margin, 0 - self.margin))

            if x.shape[-1] == 4:
                ax_clf.scatter((x[i].T[1])[mask[i]], (x[i].T[2])[mask[i]])
            else:
                ax_clf.scatter(x[i].T[0][x[i].T[0] != 0], x[i].T[1][x[i].T[0] != 0])

            ax_reg.plot(reg_pred[i][mask[i]])
            ax_reg.plot(y_reg[i][mask[i]])
            ax_reg.plot(y_reg[i][mask[i]] + self.acceptance_window, 'k--', alpha=0.5)
            ax_reg.plot(y_reg[i][mask[i]] - self.acceptance_window, 'k--', alpha=0.5)

            plt.show()

    def compare_models(self, models, data, best_of=1, mode='clf'):

        # Basic data manipulation
        x, y = data
        mask = x.sum(-1) != 0

        # Plotting anchor marks
        if mode == 'clf':
            plt.plot(np.ones(100) * 0.9, 'k--', alpha=0.2)
            plt.plot(np.ones(100) * 0.95, 'g--', alpha=0.2)
            plt.xlim((-4, 104))
            plt.ylim((-0.04, 1.04))
        if mode == 'reg':
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1e'))

        model_names = ['st-s', 'st-l', 'mt-s', 'mt-m']
        for m in models:

            # Predicting the values
            clf_pred, reg_pred, rankings = self.__make_predictions(m, x, best_of)
            hist_tot, hist_clf, hist_reg, regressor_mse = self.__compute_histogram(clf_pred, reg_pred, y, rankings,
                                                                                   mask)

            if mode == 'clf':
                plt.plot((hist_clf / hist_tot))
            if mode == 'reg':
                plt.plot((hist_reg))

        # Plotting accuracies and setting title
        title = ''.format()
        if mode == 'clf':
            title += 'Accuracy on {} - Best of {}'.format(self.stroke_dataset, best_of)
            plt.ylabel('Accuracy')
        if mode == 'reg':
            title = 'MSE regressor on {}'.format(self.stroke_dataset)
            plt.ylabel('MSE')

        # Plotting axes and title
        plt.title(title)
        plt.xlabel('Gesture completion')
        if mode == 'clf':
            plt.legend(['90%', '95%'] + model_names)
        else:
            plt.legend(model_names)
        plt.show()

    def generate_step_accuracy(self, model, data, best_of=1, steps=20, mode='clf'):
        # Basic data manipulation
        x, y = data
        print("data shape - x,y ", x.shape, y.shape)
        mask = x.sum(-1) != 0

        # Predicting the values
        clf_pred, reg_pred, rankings = self.__make_predictions(model, x, best_of)
        print('clf_pred', clf_pred.shape)
        print('reg_pred', reg_pred.shape)
        print('ranking', rankings.shape)
        hist_tot, hist_clf, hist_reg, regressor_mse = self.__compute_histogram(clf_pred, reg_pred, y, rankings, mask)

        step_size = int(hist_tot.shape[0] / steps)
        print("Generating accuracy for {} at best of {}".format(model.topology, best_of))
        means = []
        reg_errors = []
        for i in range(steps):
            mean_tot = np.mean(hist_tot[(i * step_size):((i + 1) * step_size)])
            mean_clf = np.mean(hist_clf[(i * step_size):((i + 1) * step_size)])
            mse_reg = np.mean(hist_reg[(i * step_size):((i + 1) * step_size)])
            means.append(round((mean_clf / mean_tot) * 100, 2))
            reg_errors.append("{:1.1e}".format(mse_reg))

        if mode == 'clf':
            print(str(means).replace(',', '&'))
        else:
            print((str(reg_errors).replace(',', '&')).replace('\'', ''))

    def generate_progressive_accuracy(self, model, data, plot_clf=True, plot_reg=True, best_of=1, indexToLabel=None):

        # Basic data manipulation
        x, y = data
        mask = x.sum(-1) != 0

        # Predicting the values
        clf_pred, reg_pred, rankings = self.__make_predictions(model, x, best_of)
        hist_tot, hist_clf, hist_reg, regressor_mse = self.__compute_histogram(clf_pred, reg_pred, y, rankings, mask,
                                                                               indexToLabel)

        # Plotting anchor marks
        plt.plot(np.ones(self.n_bins) * 0.9, 'k--', alpha=0.2)
        plt.plot(np.ones(self.n_bins) * 0.95, 'g--', alpha=0.2)
        plt.xlim((-1, self.n_bins + 1))
        plt.ylim((-0.04, 1.04))

        # Plotting accuracies and setting title
        title = 'Accuracy on {}'.format(self.stroke_dataset)
        if plot_clf:
            title += ' - Best of {}'.format(best_of)
            plt.plot((hist_clf / hist_tot))
        if plot_reg:
            title += ' - Window of {}'.format(self.acceptance_window)
            plt.plot((regressor_mse / hist_tot), color='orange')

        # Plotting axes and title
        plt.title(title)
        plt.xlabel('Gesture completion')
        plt.ylabel('Accuracy')
        plt.show()
        print((hist_clf / hist_tot))

    def evaluate_times(self, model, samples, raws):
        if type(model) is GestuReNN:
            model.model(samples[:1])
        else:
            exit(1)

        times = []
        densities = []
        n = samples.shape[0] // 10
        for i in range(n):
            t0 = time.clock()
            model.model(samples[i:(i + 1)])
            t1 = time.clock()
            times.append(t1 - t0)
            densities.append(raws[i].shape[0])

        mean_per_point = np.mean(np.array(times) / np.array(densities))
        mean_total = np.mean(times)

        print("Mean per point: {}".format(mean_per_point))
        title = "Mean times: {}".format("{:.3f}".format(mean_total))
        plt.title(title)
        plt.plot(times)
        plt.show()

    def make_predictions(self, model, x):
        curr_clf_pred = None
        curr_reg_pred = None
        if type(model) is GestuReNN:
            if model.topology == 'mts' or model.topology == 'mtm':
                curr_clf_pred, curr_reg_pred = model.model(x)
                # curr_clf_pred = np.argmax(curr_clf_pred, axis=2)
        return curr_clf_pred, curr_reg_pred

    # def __make_predictions(self, model, x, best_of=1):
    #     # Predicting the values
    #     clf_pred = []
    #     reg_pred = []
    #     rankings = []
    #     if type(model) is GestuReNN:
    #         if model.topology == 'mts' or model.topology == 'mtm':
    #             len_preds = x.shape[0]
    #             for i in range(self.iterations):
    #                 start = i * (len_preds // self.iterations)
    #                 end = (i + 1) * (len_preds // self.iterations)
    #                 curr_clf_pred, curr_reg_pred = model.model(x[start:end])
    #                 curr_rankings = np.argsort(curr_clf_pred, axis=2)[:, :, -best_of:]
    #                 curr_clf_pred = np.argmax(curr_clf_pred, axis=2)
    #                 if i == 0:
    #                     clf_pred = curr_clf_pred
    #                     reg_pred = curr_reg_pred
    #                     rankings = curr_rankings
    #                 else:
    #                     clf_pred = np.concatenate((clf_pred, curr_clf_pred), axis=0)
    #                     reg_pred = np.concatenate((reg_pred, curr_reg_pred), axis=0)
    #                     rankings = np.concatenate((rankings, curr_rankings), axis=0)
    #         else:
    #             clf_pred = model.classifier.predict(x)
    #             reg_pred = model.regressor.predict(x)
    #             rankings = np.argsort(clf_pred, axis=2)[:, :, -best_of:]
    #             clf_pred = np.argmax(clf_pred, axis=2)
    #     else:
    #         print('Classifier and regressor should be instances of GestuReNN.')
    #         exit(1)
    #
    #     return clf_pred, reg_pred, rankings

    def __make_predictions(self, model, x, best_of=1):
        clf_pred = []
        reg_pred = []
        rankings = []
        # Predicting the values
        if type(model) is GestuReNN_GRU or type(model) is GestuReNN:
            clf_pred, reg_pred = model.model(x)
            rankings = np.argsort(clf_pred, axis=2)[:, :, -best_of:]
            clf_pred = np.argmax(clf_pred, axis=2)
        elif type(model) is GestuReNN_mts_without_regression:
            clf_pred = model.model(x)
            rankings = np.argsort(clf_pred, axis=2)[:, :, -best_of:]
            clf_pred = np.argmax(clf_pred, axis=2)
        else:
            print('Classifier and regressor should be instances of GestuReNN.')
            exit(1)

        return clf_pred, reg_pred, rankings

    def __compute_histogram(self, clf_pred, reg_pred, ground_truth, rankings, big_mask, indexToLabel=None):

        n_predictions = clf_pred.shape[0]

        hist_tot = np.zeros(self.n_bins)
        hist_clf = np.zeros(self.n_bins)
        hist_reg = np.zeros(self.n_bins)
        regressor_mse = np.zeros(self.n_bins)

        prediction_statistics = np.zeros((len(indexToLabel), len(indexToLabel)))

        for i in range(n_predictions):
            # print("--------Sample - {}----------".format(i))

            gesture_len = (clf_pred[i][big_mask[i]]).shape[0]
            ratio = self.n_bins / gesture_len

            index = 1
            for j in range(self.n_bins):

                while j > (ratio * index):
                    index += 1
                if j == self.n_bins - 1:
                    prediction_statistics[ground_truth[i, (index - 1)], rankings[i, (index - 1), 0]] += 1
                    # print('classification gt, [pred]', ground_truth[i, (index - 1)], rankings[i, (index - 1)])
                    # print('Regression gt, pred', reg_pred[i, (index - 1)], (j / (self.n_bins - 1)))
                if ground_truth[i, (index - 1)] in rankings[i, (index - 1)]:
                    hist_clf[j] += 1
                try:
                    if abs(reg_pred[i, (index - 1)] - (j / float(self.n_bins - 1))) < self.acceptance_window:
                        hist_reg[j] += 1
                except:
                    pass

                hist_tot[j] += 1
                try:
                    reg_y = reg_pred[i, (index - 1)]
                    regressor_mse[j] += abs(
                        reg_y - (j / float(self.n_bins - 1)))  # * (reg_y - (j / float(self.n_bins-1)))
                except:
                    pass
        regressor_mse /= n_predictions
        plt.plot(regressor_mse)
        plt.show()
        self.print_prediction_statistics(prediction_statistics, indexToLabel)

        return hist_tot, hist_clf, hist_reg, regressor_mse

    def print_prediction_statistics(self, prediction_statistics, indexToLabel=None):
        print('========================================')
        if indexToLabel is None:
            for i in range(16):
                strng = "{} - ".format(i)
                for j in range(16):
                    strng += " |{}".format(prediction_statistics[i, j])
                strng += "  | {}%".format((prediction_statistics[i, i] * 100) / np.sum(prediction_statistics[i, :]))
                print(strng)
        else:
            for i, label in indexToLabel.items():
                strng = "{} ({}) - ".format(label, np.sum(prediction_statistics[i, :]))
                for j in range(len(indexToLabel)):
                    if prediction_statistics[i, j] > 0:
                        strng += "| {}: {}".format(indexToLabel[j], prediction_statistics[i, j])
                strng += "  | {}%".format((prediction_statistics[i, i] * 100) / np.sum(prediction_statistics[i, :]))
                print(strng)
