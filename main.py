from GraphicManager import GraphicManager
from GestuReNN_mts import GestuReNN
from DataLoader import DataLoader
import numpy as np
import time
import json

import tensorflow as tf

print('no. of gpus  ', tf.config.experimental.list_physical_devices("GPU"))

# test on old recognition dataset
# class Parameters:
#     pad = True
#     include_fingerup = False
#     test_size = 0.2
#     method = 'G3'
#     dataset = 'Napkin'
#     load_mode = 'test'
#     augmentFactor = 3
#     datasetFolder = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/test'
#     fileType = 'json'
#     labelJsonPath = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/test/labelDict_10_classes.json'
#     batchSize = 128
#     modelPath = '/Users/murtuza/Desktop/SketchAI/checkpoints/models/Napkin_latest_10_classes'

# test on validation set of training data
# class Parameters:
#     pad = True
#     include_fingerup = False
#     test_size = 0.2
#     method = 'G3'
#     dataset = 'Napkin'
#     load_mode = 'validation'
#     augmentFactor = 3
#     datasetFolder = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/csv'
#     fileType = 'csv'
#     labelJsonPath = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/test/labelDict_10_classes.json'
#     batchSize = 128
#     modelPath = '/Users/murtuza/Desktop/SketchAI/checkpoints/models/Napkin_latest_10_classes'


# train on Napkin dataset.
class Parameters:
    pad = True
    include_fingerup = False
    test_size = 0.2
    method = 'G3'
    dataset = 'Napkin'
    load_mode = 'train'
    augmentFactor = 3
    datasetFolder = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/csv'
    fileType = 'csv'
    labelJsonPath = '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/test/labelDict_10_classes.json'
    batchSize = 64
    modelPath = '/Users/murtuza/Desktop/SketchAI/checkpoints/models/Napkin_latest'

p = Parameters()
# Data loading
dl = DataLoader(dataset=p.dataset, load_mode=p.load_mode, labelJsonPath=p.labelJsonPath, datasetFolder=p.datasetFolder,
                fileType=p.fileType, include_fingerup=p.include_fingerup, augmentFactor=p.augmentFactor)

print(dl.validation_set_classifier[0].shape)
print(dl.train_set_classifier[0].shape)
for k, v in dl.labels_dict.items():
    print('{} - {}'.format(v, k))

# model initialization.
model_mts = GestuReNN(plot=False, labelJsonPath=p.labelJsonPath,
                      batch_size=p.batchSize, model_path=p.modelPath, include_fingerup=p.include_fingerup)
graphic_manager = GraphicManager(dataset=p.dataset, n_bins=10)

if p.load_mode == 'train':
    with open(p.labelJsonPath, 'w') as fp:
        json.dump(dl.labels_dict, fp)
    model_mts.fit_model(dl.train_set_classifier,
                        dl.validation_set_classifier,
                        dl.train_set_regressor,
                        dl.validation_set_regressor)
else:
    print("loading model ...")
    model_mts.load_model()
    print('Model Loaded!!')
    graphic_manager.generate_progressive_accuracy(model_mts, dl.test_set_classifier, plot_clf=True, plot_reg=False,
                                                  best_of=1, indexToLabel=dl.get_index_to_label())

# single input prediction
# input_curve3 = [ 34, 242, 5380, 34, 238, 5396.3623046875, 33, 180.75, 5630.5869140625, 32.5, 152.125, 5747.69921875, 32, 123.5, 5864.8115234375, 32, 35, 6226.8330078125, 30, 13, 6317.1982421875, 30, 9, 6333.5615234375, 30, 5, 6349.923828125, 28, 2, 6364.6728515625, 22, 2.25, 6389.2373046875, 19, 2.375, 6401.5205078125, 16, 2.5, 6413.802734375, 9, 1.25, 6442.890625, 5.5, 0.625, 6457.43359375, 3.75, 0.3125, 6464.7060546875, 2, 0, 6471.9775390625, 2, 2, 6480.1591796875, 0, 32, 6603.150390625, 0, 111, 6926.310546875, 2.5, 121.5, 6970.462890625, 3, 229, 7410.2119140625, 3, 236, 7438.845703125, 3, 239.5, 7453.1630859375, 3, 243, 7467.48046875, 6, 243, 7479.751953125, 14, 240, 7514.703125, 17, 240.5, 7527.1435546875, 20, 241, 7539.5849609375, 27, 241.5, 7568.29296875, 30.5, 241.75, 7582.646484375, 32.25, 241.875, 7589.8232421875, 34, 242, 7597 ]
# x = np.array(input_curve3).reshape((-1,3))
# x[:,-1] = 0
# x[-1,-1] = 1  #fingure_up with last coordinate.
# x = np.expand_dims(x, axis=0)
#
# start_time = time.time()
# clf, reg = graphic_manager.make_predictions(model_mts, x)
# print("--- %s seconds ---" % (time.time() - start_time))
# print(clf,reg)
