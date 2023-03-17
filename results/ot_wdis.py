'''A Wasserstein Subsequence Kernel for Time Series'''
'''This is one example code used in ICDM https://github.com/BorgwardtLab/WTK built upon POT'''
'''We made some modification for distance visualization'''


import numpy as np
import ot

from sklearn.metrics import pairwise
from sklearn.preprocessing import scale

from tensorflow.keras.layers import Dense, ZeroPadding1D, Reshape
from tensorflow.keras import models
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import os
from tqdm import tqdm
## time series NN models
from ts_model import AttRNN_Model, ARTLayer, WARTmodel, make_model
from ts_dataloader import readucr, plot_acc_loss
# from vggish.model import Vggish_Model
import argparse
import time as ti
# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--mod", type = int, default = 2, help = "Single input seq (0), multiple input aug (1), repro w/ TF (2)")
parser.add_argument("--net", type = int, default = 0, help = "Pretrained (0), AttRNN (#32), (1) VGGish (#512)")
parser.add_argument("--dataset", type = int, default = 0, help = "Ford-A (0), Beef (1), ECG200 (2), Wine (3), Earthquakes (4), Worms (5), Distal (6), Outline Correct (7), ECG-5k (8), ArrowH (9), CBF (10), ChlorineCon (11)")
parser.add_argument("--mapping", type= int, default=1, help = "number of multi-mapping")
parser.add_argument("--eps", type = int, default = 100, help = "Epochs") 
parser.add_argument("--per", type = int, default = 0, help = "save weight per N epochs")
parser.add_argument("--dr", type=int, default = 4, help = "drop out rate")
parser.add_argument("--seg", type=int, default = 1, help = "seg padding number")
args = parser.parse_args()


x_train, y_train, x_test, y_test = readucr(args.dataset)
    
y_train = [np.uint32(i) for i in y_train]
y_test = [np.uint32(i) for i in y_test]

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# The x_test and y_test are the official validation set in the UCR.

num_classes = len(np.unique(y_train))

if np.min(np.unique(y_train))!=0: #for those tasks with labels not start from 0, shift to 0 
    max_v = np.max(y_train)
    y_train=[i%max_v  for i in y_train]
    y_test=[i%max_v  for i in y_test]

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = np.array(y_train)[idx]

print("--- X shape : ", x_train[0].shape, "--- Num of Classes : ", num_classes) ## target class


## Pre-trained Model for Adv Program  
if args.net == 0:
    pr_model = AttRNN_Model()
elif args.net == 1: # fine-tuning with additive dense layer
    pr_model = VGGish_Model()
elif args.net == 2: # audio-set output classes  = 128
    pr_model = VGGish_Model(audioset = True)
elif args.net == 3: # unet
    pr_model = AttRNN_Model(unet= True)


# pr_model.summary()

## # of Source classes in Pre-trained Model
if args.net != 2: ## choose pre-trained network 
    source_classes = 36 ## Google Speech Commands
elif args.net == 2:
    source_classes = 128 ## AudioSet by VGGish
else:
    source_classes = 512 ## VGGish feature num

target_shape = x_train[0].shape

## Adv Program Time Series (ART)
mapping_num = args.mapping
seg_num = args.seg
drop_rate = args.dr*0.1

pr_model.summary()



try:
    assert mapping_num*num_classes <= source_classes
except AssertionError:
    print("Error: The mapping num should be smaller than source_classes / num_classes: {}".format(source_classes//num_classes)) 
    exit(1)

model = WARTmodel(target_shape, pr_model, source_classes, mapping_num, num_classes, args.mod, seg_num, drop_rate)
# else:
# model = pr_model # define for transfer learning


## Loss
adam = tf.keras.optimizers.Adam(lr=0.05,decay=0.48)
save_path = "weight/" + "beta/No" + str(args.dataset) +"_map" + str(args.mapping) + "-{epoch:02d}-{val_accuracy:.4f}.h5"
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
if args.mod == 0: # single input w/ random mapping
    y_train = keras.utils.to_categorical(y_train, source_classes)
    y_test = keras.utils.to_categorical(y_test, source_classes)
else: # with many to one mapping
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test,y_test), callbacks= exp_callback)

def comput_wd(model, kys, w_path):

    model.load_weights("weight/eps100/wdis_No"+ str(args.dataset)+"/"+w_path)
    t_score = model.evaluate(x_train, y_train, verbose=0)

    v_score = model.evaluate(x_test, y_test, verbose=0)

    # get reprogram layer output only
    rp_layer = model.get_layer(index=2)
    rp_model = models.Model([model.inputs], [rp_layer.output])
    # rp_model.summary()

    rw_test = rp_model.predict(x_test[kys,:]) # take n_dis samples
    rw_train = rp_model.predict(x_train[kys,:])

    p = args.norm 

    # Compute wasserstein distance matrices with subsequent length k=1
    D_test = transform_to_dist_matrix(rw_test[:,0], x_test[kys, :, 0], 1)
    D_train = transform_to_dist_matrix(rw_train[:,0], x_train[kys, :, 0], 1)
    # print("-- D_train shape: ", D_train.shape, " D_test shape:", D_test.shape) 
    return t_score, v_score, np.linalg.norm(D_train , ord=p), np.linalg.norm(D_test , ord=p)


max_l = min(x_test.shape[0], x_train.shape[0])
kys = np.random.choice(max_l, min(max_l, 100), replace=False)
# w_book = ['No8_map7-02-0.5844.h5',"No8_map7-04-0.5956.h5", "No8_map7-06-0.8807.h5", "No8_map7-08-0.7551.h5", "No8_map7-10-0.9198.h5"]

w_book = [f for f in os.listdir( 'weight/eps100/wdis_No'+ str(args.dataset) + '/') if f.endswith('.h5')]

tr_loss = []
val_loss = []
tr_acc = []
val_acc = []
tr_wd = []
val_wd = []

for i in tqdm(sorted(w_book)):
    model_i = WARTmodel(target_shape, pr_model, source_classes, mapping_num, num_classes)
    model_i.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    t_s, v_s, d_tr, d_te = comput_wd(model_i, kys, w_path = i)
    tr_loss.append(t_s[0])
    tr_acc.append(t_s[1])
    tr_wd.append(d_tr)
    val_loss.append(v_s[0])
    val_acc.append(v_s[1])
    val_wd.append(d_te)

    
'''A Wasserstein Subsequence Kernel for Time Series, C Bock et al.'''
'''This is one example code used in ICDM https://github.com/BorgwardtLab/WTK built upon POT, or you can use POT directly.'''   
    
def transform_to_dist_matrix(time_series_train, time_series_test, k, normalized=False):
    '''
    Computes the distance matrices for training and test.
    If time_series_train is of shape n x m,
    the resulting training matrix is of shape n x n.
    If time_series_test is of shape o x m,
    the resulting test matrix is of shape n x o.
    Args:
        time_series_train (np.ndarray): Training time series
        time_series_test (np.ndarray): Test time series
        k (int): Subsequence length
        normalized (bool): Whether to normalized subsequences or not
    Returns
        np.ndarray: Kernel matrix for training
        np.ndarray: Kernel matrix for testing
    '''

    return pairwise_subsequence_kernel(time_series_train,
        time_series_test, k, wasserstein_kernel, normalized=normalized)

def get_kernel_matrix(distance_matrix, psd=False, gamma=0.1, tol=1e-8):
    '''
    Returns a kernel matrix from a given distance matrix by calculating
        np.exp(-gamma*distance_matrix)
    Args:
        distance_matrix (np.ndarray): Square distance matrix
        psd (bool): Whether to ensure positive definiteness
        gamma (float): Gamma for kernel matrix calculation
        tol (float): Tolerance when removing negative eigenvalues
    '''
    M = np.exp(-gamma*distance_matrix)
    # Add psd-ensuring conditions
    return M if not psd else ensure_psd(M, tol=tol)

def subsequences(time_series, k):
    time_series = np.asarray(time_series)
    n = time_series.size
    shape = (n - k + 1, k)
    strides = time_series.strides * 2

    return np.lib.stride_tricks.as_strided(
        time_series,
        shape=shape,
        strides=strides
    )

def wasserstein_kernel(subsequences_1, subsequences_2, metric='euclidean'):
    '''
    Calculates the distance between two time series using their
    corresponding set of subsequences. The metric used to align
    them may be optionally changed.
    '''

    C = ot.dist(subsequences_1, subsequences_2, metric=metric)
    return ot.emd2([], [], C)

def binarized_wasserstein_kernel(subsequences_1, subsequences_2, metric='euclidean'):
    '''
    Calculates the distance between two time series using their
    corresponding set of subsequences. The metric used to align relies
    on a thresholded version of the euclidean (or other) distance
    '''

    C = ot.dist(subsequences_1, subsequences_2, metric=metric)
    C = (C>np.percentile(C,5)).astype(int)
    return ot.emd2([], [], C)

def linear_kernel(subsequences_1, subsequences_2):
    '''
    Calculates the linear kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_lin = pairwise.linear_kernel(subsequences_1, subsequences_2)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_lin)/(n*m)

def polynomial_kernel(subsequences_1, subsequences_2, p=2, c=1.0):
    '''
    Calculates the linear kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_poly = pairwise.polynomial_kernel(subsequences_1, subsequences_2,
        degree=p, coef0=c, gamma=1.0)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_poly)/(n*m)

def rbf_kernel(subsequences_1, subsequences_2):
    '''
    Calculates the rbf kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_rbf = pairwise.rbf_kernel(subsequences_1, subsequences_2)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_rbf)/(n*m)


def custom_rbf_kernel(subsequences_1, subsequences_2, e_dist, gamma):

    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    K_rbf = np.exp(-gamma*(e_dist**2))

    return np.sum(K_rbf)/(n*m)

def brownian_bridge_kernel(subsequences_1, subsequences_2, c):

    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    K_brown = c - np.abs(subsequences_1-subsequences_2)
    K_brown[K_brown<0] = 0


    return np.sum(K_brown)/(n*m)

def pairwise_subsequence_kernel(
    time_series_train,
    time_series_test,
    k,
    functor=wasserstein_kernel, 
    par_grid = [1],
    normalized = False):
    '''
    Applies a calculation functor to all pairs of a data set. As
    a result, two matrices will be calculated:
    1. The square matrix between all pairs of training samples
    2. The rectangular matrix between test samples (rows), and
       training samples (columns).
    These matrices can be fed directly to a classifier.
    Notice that this function will apply the kernel to *each* of
    the subsequences in both time series.
    '''

    n = len(time_series_train)
    m = len(time_series_test)

    K_train = np.zeros((n, n))  # Need to initialize with zeros for symmetry
    K_test = np.empty((m, n))   # Since this is rectangular, no need for zeros

    if functor in (custom_rbf_kernel, brownian_bridge_kernel):

        K_par_train = []
        K_par_test = []

        for i in range(len(par_grid)):
            K_par_train.append(np.zeros((n, n)))
            K_par_test.append(np.empty((m, n)))
    # Create subsequences of the time series. These cannot be easily
    # shared with other calls of the method.

    subsequences_train = dict()
    subsequences_test = dict()

    for i, ts_i in enumerate(time_series_train):
        subsequences_train[i] = subsequences(ts_i, k)

    for i, ts_i in enumerate(time_series_test):
        subsequences_test[i] = subsequences(ts_i, k)


    # Evaluate the functor for *all* relevant pairs, while filling up
    # the initially empty kernel matrices.

    desc = 'Pairwise kernel computations'
    # for i, ts_i in enumerate(tqdm.tqdm(time_series_train, desc=desc)):
    for i, ts_i in enumerate(time_series_train):
        for j, ts_j in enumerate(time_series_train[i:]):
            s_i = subsequences_train[i]     # first shapelet
            s_j = subsequences_train[i + j] # second shapelet
            if normalized:
                s_i = scale(s_i, axis=1)
                s_j = scale(s_j, axis=1)

            if functor in (custom_rbf_kernel, brownian_bridge_kernel):
                # euclidean distance
                if functor == custom_rbf_kernel:
                    e_dist = pairwise.euclidean_distances(s_i, s_j)

                for idx, par in enumerate(par_grid):
                    if functor == custom_rbf_kernel:
                        K_par_train[idx][i, i + j] = functor(s_i, s_j, e_dist, par)
                    else:
                        K_par_train[idx][i, i + j] = functor(s_i, s_j, par)
            else:
                K_train[i, i + j] = functor(s_i, s_j)

        for j, ts_j in enumerate(time_series_test):
            s_i = subsequences_train[i] # first shapelet

            # Second shapelet; notice that there is no index shift in
            # comparison to the code above.
            s_j = subsequences_test[j]
            if normalized:
                s_i = scale(s_i, axis=1)
                s_j = scale(s_j, axis=1)

            if functor in (custom_rbf_kernel, brownian_bridge_kernel):
                if functor == custom_rbf_kernel:
                    # euclidean distance
                    e_dist = pairwise.euclidean_distances(s_i, s_j)

                for idx, par in enumerate(par_grid):

                    if functor == custom_rbf_kernel:
                        K_par_test[idx][j, i] = functor(s_i, s_j, e_dist, par)
                    else:
                        K_par_test[idx][j, i] = functor(s_i, s_j, par)



            else:
                # Fill the test matrix; since it has different dimensions
                # than the training matrix, the indices are swapped here.
                K_test[j, i] = functor(s_i, s_j)


    # Makes the matrix symmetric since we only fill the upper diagonal
    # in the code above.


    # Makes the matrix symmetric since we only fill the upper diagonal
    # in the code above.
    if functor in (custom_rbf_kernel, brownian_bridge_kernel):
        for k_idx in range(len(par_grid)):
            K_train_cur = K_par_train[k_idx]
            K_par_train[k_idx] = K_train_cur + K_train_cur.T

        K_train = K_par_train
        K_test = K_par_test
    else:
        K_train = K_train + K_train.T


    return K_train, K_test

from numpy.linalg import eigh

def ensure_psd(K, tol=1e-8):
    # Helper function to remove negative eigenvalues
    w,v = eigh(K)
    if (w<-tol).sum() >= 1:
        neg = np.argwhere(w<-tol)
        w[neg] = 0
        Xp = v.dot(np.diag(w)).dot(v.T)
        return Xp
    else:
        return K

   
