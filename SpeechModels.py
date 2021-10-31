from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from vggish import VGGish
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

def AttRNNSpeechModel(nCategories, samplingrate=16000,
                      inputLength=16000, unet = False, rnn_func=L.LSTM):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = L.Permute((2, 1, 3))(x)
    if unet == True:
        x = L.Conv2D(16, (5, 1), activation='relu', padding='same')(x)
        up = L.BatchNormalization()(x)
        x = L.Conv2D(32, (5, 1), activation='relu', padding='same')(up)
        x = L.BatchNormalization()(x)
        x = L.Conv2D(16, (5, 1), activation='relu', padding='same')(x)
        down = L.BatchNormalization()(x)
        merge = L.Concatenate(axis=3)([up,down])
        x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(merge)
        x = L.BatchNormalization()(x)
    else:
        x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    # dot product attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def VggishModel(nCategories = 36,iLen= 16000, sr= 16000, audioset = False):

    inputs = L.Input((iLen,), name='input')

    x = L.Reshape((1, iLen))(inputs)

    if audioset == True:
        para_m = [128, 3800, 29]

    else: # Google pre-trained 
        para_m = [32, 7500, 31]
    
    m = Melspectrogram(n_dft=1024, n_hop=para_m[0], input_shape=(1, iLen),
                               padding='same', sr=sr, n_mels=64,

                               fmin=125.0, fmax=para_m[1], power_melgram=1.0,
                               return_decibel_melgram=True, trainable_fb=False,
                               trainable_kernel=False,
                               name='mel_stft') 
    m.trainable = False
    mfs = m(x)
    mfs = L.Permute((2,1,3))(mfs)
    mfs = L.Cropping2D(cropping=((0, para_m[2]), (0, 0)))(mfs)

    if audioset == True:
        vgg_model = VGGish(include_top=True, load_weights=True)
    else:
        vgg_model = VGGish(include_top=False, load_weights=False)
    
    vgg_emd = vgg_model(mfs)
    if audioset == True:
        output = vgg_emd
    else:
        output = L.Dense(nCategories, activation='softmax', name='output')(vgg_emd)
    
    model = Model(inputs=[inputs], outputs=[output], name='VGG_pre')
    # model.summary()
    return model

## Updated from 2021 March with TF Hub
import y_params as yamnet_params
import yamnet as yamnet_model

def Yamnet():

    params = yamnet_params.Params(sample_rate=16000, patch_hop_seconds=0.1)
    # Set up the YAMNet model.
    class_names = yamnet_model.class_names('weight/yamnet_class_map.csv')
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('weight/yamnet.h5')
    
    return yamnet
