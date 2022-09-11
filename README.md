# Voice2Series-Reprogramming

Voice2Series: Adversarial Reprogramming Acoustic Models for Time Series Classification


<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/img.png" width="500">


-  [Paper](https://arxiv.org/pdf/2106.09296.pdf) | [Colab Demo](https://colab.research.google.com/drive/18WpsEfz_qjjHcA7BVW-y9SHN3XLKT1Yq?usp=sharing) | [Video](https://slideslive.com/38958989/voice2series-reprogramming-acoustic-models-for-time-series-classification?ref=speaker-18620-latest) | [Slides](https://icml.cc/media/icml-2021/Slides/9059.pdf)


<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/layers.png" width="500">


- We provide an end-to-end approach (Repro. layer) to reprogram on time series data on `raw waveform` with a differential mel-spectrogram layer from kapre. 

- No offiline acoustic feature extraction and all layers are differentiable.

- updated: if you have used the `ECG 200` dataset in this code, please `git pull` and refer to the issue for [one reported label loading error](https://github.com/huckiyang/Voice2Series-Reprogramming/issues/1). (has been fixed)

### Environment 

<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />


Tensorflow 2.2 (CUDA=10.0) and Kapre 0.2.0. 

- PyTorch noted: Echo to many interests from the community, we will also provide Pytorch V2S layers and frameworks, incoperating the new torch audio layers. Feel free to email the authors for `further reprogramming collaboration`.

- option 1 (from yml)

```shell
conda env create -f V2S.yml
```

- option 2 (from clean python 3.6)

```shell
pip install tensorflow-gpu==2.1.0
pip install kapre==0.2.0
pip install h5py==2.10.0
pip install pyts
```

### Training

- Random Mapping 

Please also check the paper for actual validation details. Many Thanks!


```python
python v2s_main.py --dataset 2 --eps 20 --mod 2 --seg 6
```


- Result

```shellseg idx: 0 --> start: 0, end: 500
seg idx: 1 --> start: 5000, end: 5500
seg idx: 2 --> start: 10000, end: 10500
Tensor("AddV2_2:0", shape=(None, 16000, 1), dtype=float32)
--- Preparing Masking Matrix
Model: "model_1"
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 500, 1)]     0                                            
__________________________________________________________________________________________________
zero_padding1d (ZeroPadding1D)  (None, 16000, 1)     0           input_1[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_AddV2 (TensorFlowOp [(None, 16000, 1)]   0           zero_padding1d[0][0]             
__________________________________________________________________________________________________
zero_padding1d_1 (ZeroPadding1D (None, 16000, 1)     0           input_1[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_AddV2_1 (TensorFlow [(None, 16000, 1)]   0           tf_op_layer_AddV2[0][0]          
                                                                 zero_padding1d_1[0][0]           
__________________________________________________________________________________________________
zero_padding1d_2 (ZeroPadding1D (None, 16000, 1)     0           input_1[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_AddV2_2 (TensorFlow [(None, 16000, 1)]   0           tf_op_layer_AddV2_1[0][0]        
                                                                 zero_padding1d_2[0][0]           
__________________________________________________________________________________________________
art_layer (ARTLayer)            (None, 16000, 1)     16000       tf_op_layer_AddV2_2[0][0]        
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 16000)        0           art_layer[0][0]                  
__________________________________________________________________________________________________
model (Model)                   (None, 36)           1292911     reshape_1[0][0]                  
==================================================================================================
Total params: 1,308,911
Trainable params: 16,000
Non-trainable params: 1,292,911
__________________________________________________________________________________________________
Epoch 1/5
2021-09-21 00:39:41.269756: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-09-21 00:39:41.497716: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
113/113 [==============================] - 6s 49ms/step - loss: 5.0755 - accuracy: 0.9431 - val_loss: 3.7315 - val_accuracy: 0.9985
Epoch 2/5
113/113 [==============================] - 4s 39ms/step - loss: 3.1852 - accuracy: 0.9939 - val_loss: 2.7873 - val_accuracy: 0.9902
Epoch 3/5
113/113 [==============================] - 4s 39ms/step - loss: 2.5128 - accuracy: 0.9989 - val_loss: 2.2929 - val_accuracy: 0.9985
Epoch 4/5
113/113 [==============================] - 4s 39ms/step - loss: 2.1230 - accuracy: 0.9994 - val_loss: 1.9733 - val_accuracy: 0.9992
Epoch 5/5
113/113 [==============================] - 4s 38ms/step - loss: 1.8629 - accuracy: 0.9997 - val_loss: 1.7518 - val_accuracy: 1.0000
--- Train loss: 1.7529315948486328
- Train accuracy: 1.0
--- Test loss: 1.7516217231750488
- Test accuracy: 1.0
=== Best Val. Acc:  1.0  At Epoch of  4

```

- Many-to-one Label Mapping

```python
python v2s_main.py --dataset 0 --eps 5 --mapping 3 --mod 1
```

- Results

```shell
seg idx: 0 --> start: 0, end: 500
Tensor("AddV2:0", shape=(None, 16000, 1), dtype=float32)
--- Preparing Masking Matrix
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 500, 1)]     0                                            
__________________________________________________________________________________________________
zero_padding1d (ZeroPadding1D)  (None, 16000, 1)     0           input_1[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_AddV2 (TensorFlowOp [(None, 16000, 1)]   0           zero_padding1d[0][0]             
__________________________________________________________________________________________________
art_layer (ARTLayer)            (None, 16000, 1)     16000       tf_op_layer_AddV2[0][0]          
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 16000)        0           art_layer[0][0]                  
__________________________________________________________________________________________________
model (Model)                   (None, 36)           1292911     reshape_1[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_MatMul (TensorFlowO [(None, 6)]          0           model[1][0]                      
__________________________________________________________________________________________________
tf_op_layer_Shape (TensorFlowOp [(2,)]               0           tf_op_layer_MatMul[0][0]         
__________________________________________________________________________________________________
tf_op_layer_strided_slice (Tens [()]                 0           tf_op_layer_Shape[0][0]          
__________________________________________________________________________________________________
tf_op_layer_Reshape_2/shape (Te [(3,)]               0           tf_op_layer_strided_slice[0][0]  
__________________________________________________________________________________________________
tf_op_layer_Reshape_2 (TensorFl [(None, 2, 3)]       0           tf_op_layer_MatMul[0][0]         
                                                                 tf_op_layer_Reshape_2/shape[0][0]
__________________________________________________________________________________________________
tf_op_layer_Mean (TensorFlowOpL [(None, 2)]          0           tf_op_layer_Reshape_2[0][0]      
==================================================================================================
Total params: 1,308,911
Trainable params: 16,000
Non-trainable params: 1,292,911
__________________________________________________________________________________________________
Epoch 1/5
2021-09-21 01:23:21.163046: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-09-21 01:23:21.389418: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
113/113 [==============================] - 5s 48ms/step - loss: 2.0503 - accuracy: 1.0000 - val_loss: 1.3729 - val_accuracy: 1.0000
Epoch 2/5
113/113 [==============================] - 4s 40ms/step - loss: 1.1730 - accuracy: 1.0000 - val_loss: 1.0234 - val_accuracy: 1.0000
Epoch 3/5
113/113 [==============================] - 4s 40ms/step - loss: 0.9352 - accuracy: 1.0000 - val_loss: 0.8614 - val_accuracy: 1.0000
Epoch 4/5
113/113 [==============================] - 4s 40ms/step - loss: 0.8044 - accuracy: 1.0000 - val_loss: 0.7538 - val_accuracy: 1.0000
Epoch 5/5
113/113 [==============================] - 4s 39ms/step - loss: 0.7154 - accuracy: 1.0000 - val_loss: 0.6810 - val_accuracy: 1.0000
--- Train loss: 0.680957019329071
- Train accuracy: 1.0
--- Test loss: 0.6809701919555664
- Test accuracy: 1.0
=== Best Val. Acc:  1.0  At Epoch of  0
```


### Class Activation Mapping

```python
python cam_v2s.py --dataset 5 --weight wNo5_map6-88-0.7662.h5 --mapping 6 --layer conv2d_1
```

<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/results/0715_0318_ts_No5.png" width="600">

### Theoretical Discussion

- For sliced wasserstein distance mapping and theoretical analysis, we use the [POT](https://pythonot.github.io) package ([JMLR 2021](https://www.jmlr.org/papers/volume22/20-451/20-451.pdf)). 

<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/repro_theo.png" width="500">


- The population risk for the target task via reprogramming a K-way source neural network classifier is upper bounded by equation above.



### FAQ

- 1. Tips for tuning the model?

I would recommend using different label mapping numbers for training. For instance, you could use `--mapping 7` for `ECG 5000` dataset. The dropout rate is also an important hyperparameter for tuning the testing loss. You could use a range between `0.2` to `0.5` with `--dr 4` for `0.4` dropout rate.

- 2. Masking the target sequence is important?

V2S [mask](https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/ts_model.py#L53) is provided as an option, but the training script is not using the masking for forwarding passing. From our experiments, using or not using the masking only has small variants on the performance. This is not in conflict with the proposed theoretical analysis on learning target domain adaption.

- 3. Can we use Voice2Series for other domains or collaberate with the team?

Yes, you are welcome. Please send an email to the author for potential collaberation.


## Pre-trained models and training

- VGGish AudioSet

```bash
cd weight
pip install gdown
gdown https://drive.google.com/uc?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6

```


#### Additional Questions

Please open an issue [here](https://github.com/huckiyang/Voice2Series-Reprogramming/issues) for discussion. Thank you!
