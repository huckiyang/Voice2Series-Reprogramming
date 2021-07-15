import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from SpeechModels import AttRNNSpeechModel
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import models
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg


def multi_mapping(prob, source_num, mapping_num, target_num):
    
    mt = mapping_num * target_num ##mt must smaller than source_num
    label_map = np.zeros([source_num, mt]) ##[source_num, map_num*target_num]
    label_map[0:mt, 0:mt] = np.eye(mt) ##[source_num, map_num*target_num]
    map_prob = tf.matmul(prob, tf.constant(label_map, dtype=tf.float32)) ## [1, source_num] * [source_num, map_num*target_num] = [1, map_num*target_num]
    final_prob = tf.reduce_mean(tf.reshape(map_prob, shape=[tf.shape(map_prob)[0], target_num, mapping_num]), axis=-1) ##[target_num]
    return final_prob 


def layer_output(in_feats, model, ly_name = "batch_normalization_6 ", n = 7):
    conv_layer = model.get_layer(ly_name)
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(in_feats[n:n+1])
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    return heatmap, conv_output

def vis_map(heatmap):
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    return heatmap

def to_rgb(heatmap, h_x, w_x):
    heatmap = np.uint8(255 * vis_map((heatmap[0])))
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.flipud(np.transpose(heatmap))]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((  w_x, h_x))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Save the superimposed image
    save_path = "results/color_cam.jpg"
    superimposed_img = keras.preprocessing.image.array_to_img(jet_heatmap)
    superimposed_img.save(save_path)

    cam_img= mpimg.imread(save_path)

    return cam_img

def ts_CAM(model, x_test, y_test):
    get_last_conv = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])
    last_conv = get_last_conv([x_test[:100], 1])[0]
    get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
    softmax = get_softmax(([x_test[:100], 1]))[0]
    softmax_weight = model.get_weights()[-2]
    CAM = np.dot(last_conv, softmax_weight)
    k = 0
    # for k in range(5):
    CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
    c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
    plt.figure(figsize=(13, 7))
    plt.plot(x_test[k].squeeze())
    plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(), s=100)
    plt.title('True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))
    plt.colorbar()
    plt.savefig("cam.pdf")

