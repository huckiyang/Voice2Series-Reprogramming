# Apache Apache-2.0 License

import tensorflow as tf
import kapre
from tensorflow.keras.models import Model, load_model
from kapre.time_frequency import Melspectrogram, Spectrogram
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D, Reshape, Permute
from tensorflow.keras import initializers,regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from kapre.utils import Normalization2D
from SpeechModels import AttRNNSpeechModel, VggishModel
import numpy as np
from utils import multi_mapping
from tensorflow import keras

print("tensorflow vr. ", tf.__version__, "kapre vr. ",kapre.__version__)

def AttRNN_Model(unet= False):

    nCategs=36
    sr=16000
    #iLen=16000

    model = AttRNNSpeechModel(nCategs, samplingrate = sr, inputLength = None, unet= False)
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
    model.load_weights('weight/pr_attRNN.h5')
    # model = load_model('weight/model-attRNN.h5', custom_objects={'Melspectrogram': Melspectrogram, 'Normalization2D': Normalization2D })

    # x = np.random.rand(32,16000)
    # print(model.predict(x).shape)
    
    return model

# model.summary()

def VGGish_Model(audioset=False):
    # nCategs=36
    # iLen=16000
    model = VggishModel(audioset = audioset)
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
    if audioset == False:
        model.load_weights('weight/pr_vggish9405.h5')
    return model

# Adverserial Reprogramming layer
class ARTLayer(Layer):
    def __init__(self, tar_1ds, mod = 0, drop_rate=0.4, W_regularizer=0.05, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.l2(W_regularizer)
        self.tar_1ds = tar_1ds
        self.mod = mod
        self.dr_rate = drop_rate
        super(ARTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(16000,1),
                                      initializer=self.init, regularizer=self.W_regularizer,
                                      trainable=True)
        # Masking matrix
        print("--- Preparing Masking Matrix")
        ### pad at beginning
        if self.mod == 0:
            self.M = np.ones((16000,1)).astype('float32')
            self.M[0:self.tar_1ds,0] = 0
        
        elif self.mod == 1:
            ### pad at center
            self.M_center = np.zeros((1,self.tar_1ds)).astype('float32')
            maxlen1 = np.int(np.floor((16000-self.tar_1ds)/2) + self.tar_1ds) 
            maxlen2 = 16000
            self.pre_M = pad_sequences(self.M_center, maxlen=maxlen1, dtype='float32', padding='pre', value=1.0) #111...111[tar]
            self.pos_M = pad_sequences(self.pre_M, maxlen=maxlen2, dtype='float32', padding='post', value=1.0) #111...111[tar]111...111
        # tmp_indices = tf.where(tf.equal(self.pos_M, 0.0))
        # assert tf.reduce_sum(tmp_indices[0]) == np.int(np.floor((16000-self.tar_1ds)/2)) 
            self.M = tf.transpose(self.pos_M)

        super(ARTLayer, self).build(input_shape)  # Make layer

    def call(self, x):
        prog = K.dropout(self.W, self.dr_rate) # remove K.tanh
        out = x + prog
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], input_shape[2])

def SegZeroPadding1D(orig_x, seg_num, orig_xlen):
    src_xlen = 16000
    all_seg = src_xlen//orig_xlen
    seg_len = np.int(np.floor(all_seg//seg_num))
    aug_x = tf.zeros([src_xlen,1])
    for s in range(seg_num):
        startidx = (s*seg_len)*orig_xlen
        endidx = (s*seg_len)*orig_xlen + orig_xlen
        print('seg idx: {} --> start: {}, end: {}'.format(s, startidx, endidx))
    
        seg_x = ZeroPadding1D(padding=(startidx, src_xlen-endidx))(orig_x)
        aug_x += seg_x
    print(aug_x)
    return aug_x

# White Adversairal Reprogramming Time Series (WART) Model 
def WARTmodel(input_shape, pr_model, source_classes, mapping_num, target_classes, mod = 0, seg_num =3, drop_rate=0.4):
    x = Input(shape=input_shape)
    x_aug = SegZeroPadding1D(x, seg_num, input_shape[0])
    # x1 = ZeroPadding1D(padding=(0, 16000-input_shape[0]))(x)
    # x2 = ZeroPadding1D(padding=(16000-input_shape[0], 0))(x)
    # x3 = ZeroPadding1D(padding=(np.int(np.floor((16000-input_shape[0])/2)), np.int(np.floor((16000-input_shape[0])/2))))(x)
    # x_aug = x1 + x2 + x3
    out = ARTLayer(input_shape[0], mod = mod)(x_aug) # e.g., input_shape[0] = 500 for FordA
    out = Reshape([16000,])(out)
    probs = pr_model(out)   
    
    if mod != 0:
        probs = multi_mapping(probs, source_classes, mapping_num, target_classes)
    
    model = Model(inputs=x, outputs= probs)

    # Freezing pre-trained model
    if mod == 0:
        model.layers[-1].trainable = False
    elif mod == 1:
        model.layers[-7].trainable = False
    elif mod == 2:
        model.layers[-1].trainable = True # new setup after ICML 21, which allows reprogramming with fine-tuning.

    return model


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
