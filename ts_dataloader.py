import numpy as np
from pyts.datasets import fetch_ucr_dataset

def readucr(port = 0):

    if port == 0:
        print("--- Use FordA Dataset")
        root_tr_url = "Datasets/FordA/FordA_TRAIN.tsv"
        root_te_url = "Datasets/FordA/FordA_TEST.tsv"
    elif port == 1:
        print("--- Beef Dataset")
        root_tr_url = "Datasets/Beef/Beef_TRAIN.txt"
        root_te_url = "Datasets/Beef/Beef_TEST.txt" 
    elif port == 2:
        print("--- ECG 200 Dataset")
        root_tr_url = "Datasets/ECG200/ECG200_TRAIN.txt"
        root_te_url = "Datasets/ECG200/ECG200_TEST.txt"
    elif port == 3:
        print("--- Wine Dataset")
        root_tr_url = "Datasets/Wine/Wine_TRAIN.txt"
        root_te_url = "Datasets/Wine/Wine_TEST.txt"
    elif port == 4:
        print("--- Earthquakes Dataset")
        root_tr_url = "Datasets/Earthquakes/Earthquakes_TRAIN.txt"
        root_te_url = "Datasets/Earthquakes/Earthquakes_TEST.txt"
    elif port == 5:
        print("--- Worms Dataset")
        root_tr_url = "Datasets/Worms/Worms_TRAIN.txt"
        root_te_url = "Datasets/Worms/Worms_TEST.txt"
    elif port == 6:
        print("--- Distal Phalanx TW Dataset")
        root_tr_url = "Datasets/Distal/DistalPhalanxTW_TRAIN.txt"
        root_te_url = "Datasets/Distal/DistalPhalanxTW_TEST.txt"
    elif port == 7:
        print("--- Distal Phalanx Outline Correct Dataset")
        root_tr_url = "Datasets/DOCorrect/DistalPhalanxOutlineCorrect_TRAIN.txt"
        root_te_url = "Datasets/DOCorrect/DistalPhalanxOutlineCorrect_TEST.txt"
    elif port == 8:
        print("--- ECG 5000 Dataset")
        root_tr_url = "Datasets/ECG5000/ECG5000_TRAIN.txt"
        root_te_url = "Datasets/ECG5000/ECG5000_TEST.txt"
    elif port == 9:
        print("--- ArrowHead Dataset")
        root_tr_url = "Datasets/ArrowHead/ArrowHead_TRAIN.txt"
        root_te_url = "Datasets/ArrowHead/ArrowHead_TEST.txt"
    elif port == 10:
        print("---  Cylinder-Bell-Funnel (CBF) Dataset")
        root_tr_url = "Datasets/CBF/CBF_TRAIN.txt"
        root_te_url = "Datasets/CBF/CBF_TEST.txt"
    elif port == 11:
        print("---  ChlorineConcentration Dataset")
        root_tr_url = "Datasets/ChlorineCon/ChlorineCon_TRAIN.txt"
        root_te_url = "Datasets/ChlorineCon/ChlorineCon_TEST.txt"
    elif port > 11:
        taskname = get_taskname(port)
        print("--- "+ taskname + " Dataset")
        x_tr, x_te, y_tr, y_te = fetch_ucr_dataset(taskname, use_cache=True, data_home='Datasets/', return_X_y=True)

    if port <= 11:
        x_tr, y_tr = np_reader(root_tr_url, port)
        x_te, y_te = np_reader(root_te_url, port)
    
    return x_tr, y_tr, x_te, y_te

def get_taskname(port):
    datalist = ['FordA', 'Beef', 'ECG200', 'Wine', 'Earthquakes', 'Worms', 'DistalPhalanxTW', 'Distal Phalanx Outline', 'ECG 5000', 'ArrowHead', ' Cylinder-Bell-Funnel', 'ChlorineConcentration']
    with open ('task_list.txt', 'r') as file:
        tasklist = file.readlines()
    if port <= 11:
        taskname = datalist[port]
    elif port > 11:
        taskname = tasklist[port-12].replace('\n', '').split(' ')[1]
    return taskname

def np_reader(filename, port):

    if port == 0:
        data = np.loadtxt(filename, delimiter = "\t")
    else:
        data = np.loadtxt(filename)

    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

import matplotlib.pyplot as plt

def plot_acc_loss(x_history, eps, data_ix, map_num):

    plt.figure()
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(x_history.history["val_accuracy"], label="Val. acc")
    ax1.plot(x_history.history["accuracy"], label="Training acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(x_history.history["val_loss"], label="Val. loss")
    ax2.plot(x_history.history["loss"], label="Training loss")
    ax2.set_ylabel("Loss")
    #ax2.set_ylim(top=5.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("results/dataset_No"+ data_ix + "_eps"+ eps + "_map" + map_num + "_.png") #PadCenter/
