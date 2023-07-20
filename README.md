# Voice2Series-Reprogramming

Voice2Series: Adversarial Reprogramming Acoustic Models for Time Series Classification

<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/img.png" width="500">


-  [Paper](https://arxiv.org/pdf/2106.09296.pdf) | [Colab Demo](https://colab.research.google.com/drive/18WpsEfz_qjjHcA7BVW-y9SHN3XLKT1Yq?usp=sharing) | [Video](https://slideslive.com/38958989/voice2series-reprogramming-acoustic-models-for-time-series-classification?ref=speaker-18620-latest) | [Slides](https://icml.cc/media/icml-2021/Slides/9059.pdf)


<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/layers.png" width="500">


- We provide an end-to-end approach (Repro. layer) to reprogram on time series data on `raw waveform` with a differential mel-spectrogram layer from kapre. 

- No offiline acoustic feature extraction and all layers are differentiable.

- Pytorch version of reprogram layer could be found out in [ICASSP 23 Music Reprogramming](https://github.com/biboamy/music-repro).

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
python v2s_main.py --dataset 0 --eps 20 --mod 2 --seg 18 --mapping 1
```


- Result

```
Epoch 14/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4493 - accuracy: 0.9239 - val_loss: 0.4571 - val_accuracy: 0.9106
Epoch 15/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4297 - accuracy: 0.9306 - val_loss: 0.4381 - val_accuracy: 0.9265
Epoch 16/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4182 - accuracy: 0.9247 - val_loss: 0.4204 - val_accuracy: 0.9205
Epoch 17/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3972 - accuracy: 0.9320 - val_loss: 0.4072 - val_accuracy: 0.9242
Epoch 18/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3905 - accuracy: 0.9303 - val_loss: 0.4099 - val_accuracy: 0.9242
Epoch 19/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3765 - accuracy: 0.9320 - val_loss: 0.3924 - val_accuracy: 0.9258
Epoch 20/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3704 - accuracy: 0.9300 - val_loss: 0.3816 - val_accuracy: 0.9250
--- Train loss: 0.36046191089949786
- Train accuracy: 0.93113023
--- Test loss: 0.38329164963780027
- Test accuracy: 0.925
=== Best Val. Acc:  0.92651516  At Epoch of  14

```

- Many-to-one Label Mapping

```python
python v2s_main.py --dataset 0 --eps 20 --mod 2 --seg 18 --mapping 18
```

- Results

```shell
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.8762 - accuracy: 0.9231 - val_loss: 0.8479 - val_accuracy: 0.9182
Epoch 12/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.8360 - accuracy: 0.9236 - val_loss: 0.8191 - val_accuracy: 0.9152
Epoch 13/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7920 - accuracy: 0.9242 - val_loss: 0.7693 - val_accuracy: 0.9273
Epoch 14/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7586 - accuracy: 0.9228 - val_loss: 0.7358 - val_accuracy: 0.9235
Epoch 15/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7265 - accuracy: 0.9270 - val_loss: 0.7076 - val_accuracy: 0.9205
Epoch 16/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6980 - accuracy: 0.9247 - val_loss: 0.6707 - val_accuracy: 0.9295
Epoch 17/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6650 - accuracy: 0.9281 - val_loss: 0.6473 - val_accuracy: 0.9250
Epoch 18/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6444 - accuracy: 0.9286 - val_loss: 0.6270 - val_accuracy: 0.9303
Epoch 19/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6194 - accuracy: 0.9286 - val_loss: 0.6020 - val_accuracy: 0.9318
Epoch 20/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.5964 - accuracy: 0.9275 - val_loss: 0.5813 - val_accuracy: 0.9227
--- Train loss: 0.5795955053139845
- Train accuracy: 0.93113023
--- Test loss: 0.5856682072986256
- Test accuracy: 0.92651516
=== Best Val. Acc:  0.9318182  At Epoch of  18
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
