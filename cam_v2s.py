# Original CAM Code is modified from Yang et al. ICASSP 2021 (https://arxiv.org/pdf/2010.13309.pdf)
# Please consider to cite both de Andrade et al. 2018 and Yang et al. 2021 ICML, if you use the attention heads and CAM visualization.

from ts_model import  AttRNN_Model, ARTLayer, WARTmodel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D
import matplotlib.pyplot as plt
import time as ti
from utils import layer_output, to_rgb
import librosa
import librosa.display
from ts_dataloader import readucr
import argparse
from PIL import Image
data_ix = ti.strftime("%m%d_%H%M")

base_model = AttRNN_Model()
base_model.summary()

audios = np.load("Datasets/val_audios.npy") # load wavs files
cmds = np.load("Datasets/val_cmds.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = int, default = 5, help = "Ford-A (0), Beef (1), ECG200 (2), Wine (3)") 
parser.add_argument("--weight", type = str, default = "wNo5_map6-88-0.7662.h5", help = "weight in /weights/")
parser.add_argument("--mapping", type= int, default= 6, help = "number of multi-mapping")
parser.add_argument("--layer", type = str, default = "conv2d_1", help = "the layer for cam")
args = parser.parse_args()

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
                                          model.get_layer('mel_stft').output])


x_train, y_train, x_test, y_test = readucr(args.dataset) # 4 - Earthquake // 8 - ECG 5k
tmp_xt = x_test
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
num_classes = len(np.unique(y_train))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

art_model = WARTmodel(target_shape, model, 36,  args.mapping, num_classes, mod=2)
art_model.load_weights("weight/"+ args.weight)


ReproM = Model(inputs=art_model.input, outputs=[art_model.get_layer('reshape_1').output])

repros = ReproM.predict(x_test)


def visual_sp(audios, use='base', clayer = args.layer):

    outs, attW, specs = attM.predict(audios)

    w_x, h_x = specs[idAudio,:,:,0].shape
    i_heatmap1, _ = layer_output(audios, base_model, 'conv2d', idAudio)
#    i_heatmap2, _ = layer_output(audios, base_model, 'conv2d_1', idAudio)
    i_cam1 = to_rgb(i_heatmap1, w_x, h_x)
#    i_cam2 = to_rgb(i_heatmap2, w_x, h_x)


    plt.figure()
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2,ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    # ax1.set_title('Raw waveform', fontsize=18)
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.set_xlabel('Sample index', fontsize=18)
    ax1.plot(audios[idAudio], 'b-',label = "Reprogrammed time series")
    if use != 'base':
        x_tmp = tmp_xt[idAudio].reshape((tmp_xt[idAudio].shape[0], 1))  
        x_tmp = tf.expand_dims(x_tmp, axis=0)
        print(x_tmp.shape)
        aug_tmp = SegZeroPadding1D(x_tmp, 3, tmp_xt[idAudio].shape[0])
        ax1.plot(tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1), 'k-', label="Target time series")
        print(aug_tmp.shape)
    ax1.legend(fancybox=True, framealpha=1,  borderpad=1, fontsize=16)

    # ax2.set_title('Attention weights (log)', fontsize=18)
    ax2.set_ylabel('Log of attention weight', fontsize=18)
    ax2.set_xlabel('Mel-spectrogram index', fontsize=18)
    ax2.plot(np.log(attW[idAudio]), 'r-')

    # ax3.imshow(librosa.power_to_db(specs[idAudio,:,:,0], ref=np.max))
    img3 = ax3.pcolormesh(specs[idAudio,:,:,0])
#    plt.colorbar(img3)
    # ax3.set_title('Spectrogram visualization', fontsize=18)
    ax3.set_ylabel('Frequency', fontsize=18)
    ax3.set_xlabel('Time', fontsize=18)

    img4 = ax4.imshow(i_cam1, aspect="auto")
#    plt.colorbar(img4)
    # ax4.set_title('Class Activation Mapping Conv2d', fontsize=18)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
#    img5 = ax5.imshow(i_cam2, aspect="auto")
#    plt.colorbar(img5)
    #ax5.set_title('Class Activation Mapping Conv2d_1', fontsize=18)
#    ax5.set_xticks([])
#    ax5.set_yticks([])

    plt.tight_layout()
    if use == 'base':
        plt.savefig("results/" + data_ix + "_sp_" + cmd_k +".png")
    else:
        plt.savefig("results/" + data_ix + "_ts_No"+ str(args.dataset) +".png")
    

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

visual_sp(repros, "adv")

