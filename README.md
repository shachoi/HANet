# HANet: Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> CVPR 2020<br>
> Title: Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks<br>
> Authors: Sungha Choi (LGE, Korea Univ.), Joanne T. Kim (LLNL, Korea Univ.), Jaegul Choo (KAIST)<br>
> Paper : https://arxiv.org/abs/2003.05128 <br>

## 1. Concept Video
Click the figure to watch the youtube video of our paper!
<p align="center">
  <a href="https://www.youtube.com/watch?v=0Orj3AUfu9Y"><img src="assets/youtube_capture_p.png" alt="Youtube Video"></a><br>
</p>

## 2. Teaser Image
HANet is an add-on module for urban-scene segmentation to exploit the structural priors existing in urban-scene. It is effective and wide applicable!
<p align="center">
  <img src="assets/6529-teaser.gif" />
</p>

## 3. Benchmark
| Models | mIoU | External Link |
|:--------:|:--------:|:--------:|
| HANet (ResNext-101) | 83.2% | [Cityscapes benchmark](https://www.cityscapes-dataset.com/anonymous-results/?id=9a8b7333dcb66360b4f38ba00db7c84e7997f7203084bf6e92ca9bbabbc34640) |
| HANet (ResNet-101) | 82.1% | [Cityscapes benchmark](https://www.cityscapes-dataset.com/anonymous-results/?id=f96818d678c67c82449323203d144e530fb66102a5b5a101f599a96cc62458e7) |


## 4. Pytorch Implementation
### 4.1. Installation
Clone this repository.
```
git clone https://github.com/shachoi/HANet.git
cd HANet
```
Install following packages.
```
conda create --name hanet python=3.6
conda activate hanet
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
conda install scipy==1.4.1
conda install tqdm==4.46.0
conda install scikit-image==0.16.2
pip install tensorboardX==2.0
pip install thop
```
We tested pytorch v1.2~1.4.

### 4.2. Datasets
We evaludated HANet on [Cityscapes](https://www.cityscapes-dataset.com/) and [BDD-100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/).
> For Cityscapes, download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from https://www.cityscapes-dataset.com/downloads/<br>
> After unzipping above files, the folder structures are as follows.<br>
```
cityscapes
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```
> You should modify the following dataset path in "<path_to_hanet>/config.py" according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = '/home/nas_datasets/segmentation/cityscapes'
```
> Additionally, you can use Cityscapes coarse dataset to get best mIoU score.

### 4.3. Pretrained Models
#### 4.3.1. All models trained for our paper
You can download all models evaluated in our paper.
https://drive.google.com/drive/folders/1qetciC7G29Gg4iKHLWhCioSdMbmYeb0Y?usp=sharing

#### 4.3.2. ImageNet pretrained ResNet-101 which has three 3×3 convolutions in the first layer
> To train ResNet-101 based HANet, you should download ImageNet pretrained ResNet-101 from [this link](https://drive.google.com/file/d/1jMx3HdVqSlpIYIyG3VPi8q-ZiclOHlc7/view?usp=sharing). Put it into following directory.
```
<path_to_hanet>/pretrained/resnet101-imagenet.pth
```
> This pretrained model is from http://sceneparsing.csail.mit.edu/

#### 4.3.3. Mapillary pretrained ResNext-101
> You can finetune HANet from Mapillary pretrained ResNext-101.<br>
> Download from [this link](https://drive.google.com/file/d/1GJ4VOSiLwNuyqOgRqQoe9FbvnklI2TYe/view?usp=sharing) and put it into following directory.
```
<path_to_hanet>/pretrained/resnext_mapillary_0.47475.pth
```
### 4.4. Training Networks
According to the specification of your gpu system, you may modify the training script.
```
     python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py \  
      ...
      --bs_mult NUM_BATCH_PER_SINGLE_GPU \
```
You can train HANet (based on ResNet-101) using finely annotated training and validation set with following command.
```
<path_to_hanet>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r101_os8_hanet_best.sh
```
Additioanlly, we provide various training scripts.
The results will be stored in "<path_to_hanet>/logs/"


### 4.5. Inference
```
<path_to_hanet>$ CUDA_VISIBLE_DEVICES=0 ./scripts/eval_r101_os8.sh <weight_file_location> <result_save_location>
```
### 4.6. Cityscapes Benchmark
```
<path_to_hanet>$ CUDA_VISIBLE_DEVICES=0 ./scripts/submit_r101_os8.sh <weight_file_location> <result_save_location>
```
## 5. Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{choi2020cars,
  title={Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks},
  author={Choi, Sungha and Kim, Joanne T and Choo, Jaegul},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
### 6. Acknowledgments
Our pytorch implementation is heavily derived from [NVIDIA segmentation](https://github.com/NVIDIA/semantic-segmentation).
Thanks to the NVIDIA implementations.
