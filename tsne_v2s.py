import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D
import matplotlib.pyplot as plt
import time as ti
from utils import layer_output, to_rgb
import librosa
import librosa.display
import argparse
from PIL import Image
from ts_model import AttRNN_Model, ARTLayer, WARTmodel
import pandas as pd
from tensorflow.keras import layers as L

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

base_model = AttRNN_Model()

from ts_dataloader import readucr

audios = np.load("Datasets/val_audios.npy") # load wavs files
cmds = np.load("Datasets/val_cmds.npy")



idAudio = 0
GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11,
            'zero': 12,
            'one': 13,
            'two': 14,
            'three': 15,
            'four': 16,
            'five': 17,
            'six': 18,
            'seven': 19,
            'eight': 20,
            'nine': 1,
            'backward': 21,
            'bed': 22,
            'bird': 23,
            'cat': 24,
            'dog': 25,
            'follow': 26,
            'forward': 27,
            'happy': 28,
            'house': 29,
            'learn': 30,
            'marvin': 31,
            'sheila': 32,
            'tree': 33,
            'visual': 34,
            'wow': 35}


key_list = list(GSCmdV2Categs.keys())
cmd_k = key_list[cmds[idAudio]]
print("Input Speech Cmd: ", cmd_k)

model = base_model
attM = Model(inputs=model.input, outputs=[model.get_layer('output').output,
                                          model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output,
                                          model.get_layer('dense_1').output])

task_num = 63
path = 'weight/v2s_no63_map18_seg10_dr2_273_0.9757.h5'
x_train, y_train, x_test, y_test = readucr(task_num)
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
num_classes = len(np.unique(y_train))
seg_num = 10
mapping_num = 18
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

art_model = WARTmodel(target_shape, model, 36,  mapping_num, num_classes, mod=2)
art_model.load_weights(path)

ReproM = Model(inputs=art_model.input, outputs=[art_model.get_layer('reshape_1').output])

repros = ReproM.predict(x_test)

def transfer_att(nCategories, samplingrate = 16000, inputLength=16000):

    attMM = Model(inputs=base_model.input, outputs=[base_model.get_layer('bidirectional_1').output, base_model.get_layer('attSoftmax').output])
    sr = samplingrate
    iLen = inputLength

    inputs = L.Input((inputLength,), name='input')
    x = L.Reshape((1, -1))(inputs)
    #x = matt(x)
    attScores, bi_x = attMM(x)
    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, bi_x])  # [b_s, vec_dim]
    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)
    y_out = L.Dense(nCategories, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[y_out])

    return model

def SegZeroPadding1D(orig_x, seg_num, orig_xlen):

    src_xlen = 16000
    all_seg = src_xlen//orig_xlen
    assert seg_num <= all_seg
    seg_len = np.int(np.floor(all_seg//seg_num))
    aug_x = tf.zeros([src_xlen,1])
    for s in range(seg_num):
        startidx = (s*seg_len)*orig_xlen
        endidx = (s*seg_len)*orig_xlen + orig_xlen
        # print('seg idx: {} --> start: {}, end: {}'.format(s, startidx, endidx))
        seg_x = ZeroPadding1D(padding=(startidx, src_xlen-endidx))(orig_x)
        aug_x += seg_x

    return aug_x

tr_model = transfer_att(nCategories = num_classes)
tr_model.load_weights('weight/trsf_no63_map1_seg1_dr4_01_0.6432.h5')
trM = Model(inputs=tr_model.input, outputs=[tr_model.get_layer('dense_4').output])
# tr_model.summary()

def visual_tsne(adv_audios, origs_audio, y, num_classes, seg_num, use='base', ppl=40):

    aug_x = []
    seed_id = 1
    for i in range(origs_audio.shape[0]):
        aug_audios = SegZeroPadding1D(tf.expand_dims(origs_audio[i], axis=0), seg_num,  origs_audio.shape[1])
        aug_x.append(tf.squeeze(aug_audios, axis=0))

    aug_x = tf.stack(tf.squeeze(aug_x, axis=-1), axis=0)
    out0, attW0, specs0, dense0 = attM.predict(tf.stack(aug_x, axis=0))
    out1, attW1, specs1, dense1 = attM.predict(adv_audios)
    out2 =  trM.predict(tf.stack(aug_x, axis=0))

    print(aug_x.shape)
    print(adv_audios.shape)
    print('orig shape: ', dense0.shape)
    print('repro shape: ', dense0.shape)
    print('transfer shape: ', out2.shape)

    feat_cols = [ 'pixel'+str(i) for i in range(dense0.shape[1]) ]
    df = pd.DataFrame(dense0,columns=feat_cols)
    df['y'] = np.array(y,dtype=np.uint8)
    df['label'] = df['y'].apply(lambda i: str(i))

    # For reproducability of the results
    np.random.seed(seed_id)
    rndperm = np.random.permutation(df.shape[0])

    N = 10000
    df_subset1 = df.loc[rndperm[:N],:].copy()
    data_subset1 = df_subset1[feat_cols].values

    tsne = TSNE(n_components=2, verbose=1, perplexity=ppl, n_iter=350)
    tsne_results1 = tsne.fit_transform(data_subset1)

    df_subset1['tsne-2d-one'] = tsne_results1[:,0]
    df_subset1['tsne-2d-two'] = tsne_results1[:,1]

    #######################################################################################


    feat_cols = [ 'pixel'+str(i) for i in range(out2.shape[1]) ]
    df = pd.DataFrame(out2,columns=feat_cols)
    df['y'] = np.array(y,dtype=np.uint8)
    df['label'] = df['y'].apply(lambda i: str(i))

    # For reproducability of the results
    np.random.seed(seed_id)
    rndperm = np.random.permutation(df.shape[0])

    N = 10000
    df_subset3 = df.loc[rndperm[:N],:].copy()
    data_subset3 = df_subset3[feat_cols].values

    tsne = TSNE(n_components=2, verbose=1, perplexity=ppl, n_iter=350)
    tsne_results3 = tsne.fit_transform(data_subset3)

    df_subset3['tsne-2d-one'] = tsne_results3[:,0]
    df_subset3['tsne-2d-two'] = tsne_results3[:,1]

    #######################################################################################

    feat_cols = [ 'pixel'+str(i) for i in range(dense1.shape[1]) ]
    df = pd.DataFrame(dense1,columns=feat_cols)
    df['y'] = np.array(y,dtype=np.uint8)
    df['label'] = df['y'].apply(lambda i: str(i))

    # For reproducability of the results
    np.random.seed(seed_id)
    rndperm = np.random.permutation(df.shape[0])

    N = 10000
    df_subset2 = df.loc[rndperm[:N],:].copy()
    data_subset2 = df_subset2[feat_cols].values

    tsne = TSNE(n_components=2, verbose=1, perplexity=ppl, n_iter=350)
    tsne_results = tsne.fit_transform(data_subset2)

    df_subset2['tsne-2d-one'] = tsne_results[:,0]
    df_subset2['tsne-2d-two'] = tsne_results[:,1]

    #######################################################################################

    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.title('Before V2S reprogramming', fontsize=18)

    plt.subplots_adjust(wspace = 0.15)
    plt.grid(False)
    sns_plot1 = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("Set1", n_colors=num_classes),
        data=df_subset1,
        legend='full'
        #alpha=0.4
    )
    sns_plot1.set(xlabel='', ylabel='',xticklabels=[], yticklabels=[])
    sns_plot1.legend(loc='upper right', fontsize=16)

    plt.subplot(1, 3, 2)
    plt.title('Fine-tuned Transfer Learning', fontsize=18)
    plt.subplots_adjust(wspace = 0.15)
    plt.grid(False)
    sns_plot2 = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("Set1", n_colors=num_classes),
        data=df_subset3
        #alpha=0.4
    )
    sns_plot2.set(xlabel='', ylabel='',xticklabels=[], yticklabels=[])
    sns_plot2.legend(loc='upper right', fontsize=16)

    plt.subplot(1, 3, 3)
    plt.title('After V2S reprogramming', fontsize=18)
    plt.subplots_adjust(wspace = 0.15)
    plt.grid(False)
    sns_plot3 = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("Set1", n_colors=num_classes),
        data=df_subset2
       # alpha=0.4
    )
    sns_plot3.set(xlabel='', ylabel='',xticklabels=[], yticklabels=[])
    sns_plot3.legend(loc='upper right',fontsize=16)

    plt.tight_layout()
    plt.savefig("results/newtsne2d_task" + str(task_num) + ".pdf")

visual_tsne(repros, x_test, y_test, num_classes, seg_num, "adv", 50)
