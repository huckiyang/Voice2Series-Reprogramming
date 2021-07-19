# Voice2Series-Reprogramming

Voice2Series: Reprogramming Acoustic Models for Time Series Classification


<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/img.png" width="500">


- International Conference on Machine Learning (ICML), 2021 | [Paper](http://proceedings.mlr.press/v139/yang21j/yang21j.pdf)


### Environment


tensorflow 2.2 (CUDA=10.0) and kapre 0.2.0

- option 1 (from yml)

```shell
conda env create -f V2S.yml
```

- option 2 (from clean python 3.6)

```shell
conda install -c anaconda tensorflow-gpu
conda install -c conda-forge opencv
conda install -c anaconda pillow
```

### Training

```python
python v2s_main.py --dataset 0 --eps 100 --mapping 3
```

- Result

```shell
seg idx: 0 --> start: 0, end: 500
seg idx: 1 --> start: 5000, end: 5500
seg idx: 2 --> start: 10000, end: 10500
Tensor("AddV2_2:0", shape=(None, 16000, 1), dtype=float32)
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
Trainable params: 217,225
Non-trainable params: 1,091,686
__________________________________________________________________________________________________
Epoch 1/100
2021-07-19 01:43:32.690913: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-07-19 01:43:32.919343: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
113/113 [==============================] - 6s 50ms/step - loss: 0.0811 - accuracy: 1.0000 - val_loss: 1.5589e-04 - val_accuracy: 1.0000
Epoch 2/100
113/113 [==============================] - 5s 41ms/step - loss: 5.0098e-05 - accuracy: 1.0000 - val_loss: 1.0906e-05 - val_accuracy: 1.0000
```


### Class Activation Mapping

```python
python cam_v2s.py --dataset 5 --weight wNo5_map6-88-0.7662.h5 --mapping 6 --layer conv2d_1
```

  
 
<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/results/0715_0318_ts_No5.png" width="600">

### Reference

- Voice2Series: Reprogramming Acoustic Models for Time Series Classification


```bib

@InProceedings{pmlr-v139-yang21j,
  title = 	 {Voice2Series: Reprogramming Acoustic Models for Time Series Classification},
  author =       {Yang, Chao-Han Huck and Tsai, Yun-Yun and Chen, Pin-Yu},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {11808--11819},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/yang21j/yang21j.pdf},
  url = 	 {http://proceedings.mlr.press/v139/yang21j.html},
}

```
