# Voice2Series-Reprogramming
Voice2Series: Reprogramming Acoustic Models for Time Series Classification


<img src="https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/img/img.png" width="500">


- International Conference on Machine Learning (ICML), 2021


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

### Class Activation Mapping

```python
python cam_v2s.py --dataset 5 --weight wNo5_map6-88-0.7662.h5 --mapping 6 --layer conv2d_1
```
