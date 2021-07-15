## Author C.-H. Huck Yang, MIT License
## Copyright (c) 2021 huckiyang
## Refer to "Voice2Series: Reprogramming Acoustic Models for Time Series Classification" ICML 2021 for Training Details 
## Please Consider to Credit the related references, if you use this implementation and find this work helps. 


from tensorflow.keras.layers import Dense, ZeroPadding1D, Reshape
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
## time series NN models
from ts_model import AttRNN_Model, VGGish_Model, ARTLayer, WARTmodel, make_model
from ts_dataloader import readucr, plot_acc_loss
from resnet import transfer_res, transfer_att
# from vggish.model import Vggish_Model
import argparse
import os
import re 
# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()
# K.set_learning_phase(0)

parser = argparse.ArgumentParser()
parser.add_argument("--net", type = str, default = 'attrsf', help = "att, vggish, aset")
parser.add_argument("--dataset", type = int, default = 0, help = "Ford-A (0), Beef (1), ECG200 (2), Wine (3), Earthquakes (4), Worms (5), Distal (6), Outline Correct (7), ECG-5k (8), ArrowH (9), CBF (10), ChlorineCon (11)")
parser.add_argument("--mapping", type= int, default=1, help = "number of multi-mapping")
parser.add_argument("--eps", type = int, default = 100, help = "Epochs") 
parser.add_argument("--per", type = int, default = 0, help = "save weight per N epochs")
parser.add_argument("--dr", type=int, default = 4, help = "drop out rate")
parser.add_argument("--seg", type=int, default = 1, help = "seg padding number")
args = parser.parse_args()


x_train, y_train, x_test, y_test = readucr(args.dataset)

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))

if np.min(np.unique(y_train))!=0: #for those tasks with labels not start from 0, shift to 0 
    y_train%=num_classes 
    y_test%=num_classes

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

print("--- X shape : ", x_train[0].shape, "--- Num of Classes : ", num_classes) ## target class


## Pre-trained Model for Adv Program  
if args.net == 'att':
    pr_model = AttRNN_Model()
elif args.net == 'vggish':
    pr_model = VGGish_Model()
elif args.net == 'aset':
    pr_model = VGGish_Model(audioset = True)
elif args.net == 'unet':
    pr_model = AttRNN_Model(unet= True)

## # of Source classes in Pre-trained Model
if args.net != 'aset': ## choose pre-trained network 
    source_classes = 36 ## Google Speech Commands
else:
    source_classes = 128 ## AudioSet by VGGish

target_shape = x_train[0].shape

## Adv Program Time Series (ART)
mapping_num = args.mapping
seg_num = args.seg
drop_rate = args.dr*0.1

try:
    assert mapping_num*num_classes <= source_classes
except AssertionError:
    print("Error: The mapping num should be smaller than source_classes / num_classes: {}".format(source_classes//num_classes)) 
    exit(1)
if args.net == "trsf" or "attrsf":
    model = pr_model # already define for transfer learning
else:
    model = WARTmodel(target_shape, pr_model, source_classes, mapping_num, num_classes, seg_num, drop_rate)

## Loss
adam = tf.keras.optimizers.Adam(lr=0.05,decay=0.48)
save_path = "weight/" + "transfer/No" + str(args.dataset) +"/map" + str(args.mapping) + "_seg" + str(args.seg) + "_dr" + str(args.dr) +"_{epoch:02d}_{val_accuracy:.4f}.h5"
if args.per!= 0:
    checkpoints = tf.keras.callbacks.ModelCheckpoint(save_path, save_weights_only=True, period=args.per)
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500), checkpoints]
else:
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500)]


model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

model.summary()

batch_size = 32
epochs = args.eps

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test,y_test), callbacks= exp_callback)

score = model.evaluate(x_train, y_train, verbose=0)
print('--- Train loss:', score[0])
print('- Train accuracy:', score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])


print("=== Best Val. Acc: ", max(exp_history.history['val_accuracy']), " At Epoch of ", np.argmax(exp_history.history['val_accuracy']))
print("val. acc: ", exp_history.history['val_accuracy'])
curr_dir='weight/transfer/No' + str(args.dataset) + '/'
plot_acc_loss(exp_history, str(args.eps), str(args.dataset), str(args.mapping), str(args.seg), str(args.dr), args.net)

