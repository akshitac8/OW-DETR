# OW-DETR: Open-world Detection Transformer (CVPR 2022)

[`Paper`](https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.pdf) [`Video`](https://www.youtube.com/watch?v=saO8RHCpnaY) [`slides`](https://docs.google.com/presentation/d/1I1OyoRbKqvwB_dSLM8ybSXrB74crPX2a9R9yyWvABDc/edit?usp=sharing) [`summary slide`](https://docs.google.com/presentation/d/1zABTrvkaYlqb7u6xWv1JPIHFdsAyRkAggnmj33kuwsE/edit?usp=sharing)

#### [Akshita Gupta](https://akshitac8.github.io/)<sup>\*</sup>, [Sanath Narayan](https://sites.google.com/view/sanath-narayan)<sup>\*</sup>, [K J Joseph](https://josephkj.in), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) ####

(:star2: denotes equal contribution)

# Introduction

 Open-world object detection (OWOD) is a challenging computer vision problem, where the task is to detect a known set of object categories while simultaneously identifying unknown objects. Additionally, the model must incrementally learn new classes that become known in the next training episodes. Distinct from standard object detection, the OWOD setting poses significant challenges for generating quality candidate proposals on potentially unknown objects, separating the unknown objects from the background and detecting diverse unknown objects. Here, we introduce a novel end-to-end transformer-based  framework, OW-DETR, for open-world object detection. The proposed OW-DETR comprises three dedicated components namely, attention-driven pseudo-labeling, novelty classification and objectness scoring  to explicitly address the aforementioned OWOD challenges. Our OW-DETR explicitly encodes multi-scale contextual information, possesses less inductive bias, enables knowledge transfer from known classes to the unknown class and can better discriminate between unknown objects and background. Comprehensive experiments are performed on two benchmarks: MS-COCO and PASCAL VOC. The extensive ablations reveal the merits of our proposed contributions. Further, our model outperforms the recently introduced OWOD approach, ORE, with absolute gains ranging from  $1.8\%$ to $3.3\%$ in terms of unknown recall on MS-COCO. In the case of incremental object detection, OW-DETR outperforms the state-of-the-art for all settings on PASCAL VOC.
<br>

<p align="center" ><img width='350' src = "https://imgur.com/KXDXiAB.png"></p> 

<br>

<p align="center" ><img width='500' src = "https://imgur.com/cyeMXuh.png"></p> 





# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n owdetr python=3.7 pip
conda activate owdetr
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Backbone features

Download the self-supervised backbone from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


# Dataset & Results

### OWOD proposed splits
<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

The splits are present inside `data/VOC2007/OWOD/ImageSets/` folder. The remaining dataset can be downloaded using this [link](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k?usp=sharing)

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">4.9</td>
        <td align="center">56.0</td>
        <td align="center">2.9</td>
        <td align="center">39.4</td>
        <td align="center">3.9</td>
        <td align="center">29.7</td>
        <td align="center">25.3</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">7.5</td>
        <td align="center">59.2</td>
        <td align="center">6.2</td>
        <td align="center">42.9</td>
        <td align="center">5.7</td>
        <td align="center">30.8</td>
        <td align="center">27.8</td>
    </tr>
</table>



### Our proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

#### Dataset Preparation

The splits are present inside `data/VOC2007/OWDETR/ImageSets/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/VOC2007/OWDETR/JPEGImages/
mkdir data/VOC2007/OWDETR/Annotations/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
OW-DETR/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
```
cd OW-DETR/data
mv data/coco/train2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/OWDETR/JPEGImages/.
```
5. Use the code `coco2voc.py` for converting json annotations to xml files.

The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWDETR/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```


Currently, Dataloader and Evaluator followed for OW-DETR is in VOC format.

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">1.5</td>
        <td align="center">61.4</td>
        <td align="center">3.9</td>
        <td align="center">40.6</td>
        <td align="center">3.6</td>
        <td align="center">33.7</td>
        <td align="center">31.8</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">5.7</td>
        <td align="center">71.5</td>
        <td align="center">6.2</td>
        <td align="center">43.8</td>
        <td align="center">6.9</td>
        <td align="center">38.5</td>
        <td align="center">33.1</td>
    </tr>
</table>

    
# Training

#### Training on single node

To train OW-DETR on a single node with 8 GPUS, run
```bash
./run.sh
```

#### Training on slurm cluster

To train OW-DETR on a slurm cluster having 2 nodes with 8 GPUS each, run
```bash
sbatch run_slurm.sh
```

# Evaluation

For reproducing any of the above mentioned results please run the `run_eval.sh` file and add pretrained weights accordingly.


**Note:**
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


# Citation

If you use OW-DETR, please consider citing:

    @inproceedings{gupta2021ow,
        title={OW-DETR: Open-world Detection Transformer}, 
        author={Gupta, Akshita and Narayan, Sanath and Joseph, KJ and 
        Khan, Salman and Khan, Fahad Shahbaz and Shah, Mubarak},
        booktitle={CVPR},
        year={2022}
    }

# Contact

Should you have any question, please contact :e-mail: akshita.sem.iitr@gmail.com

**Acknowledgments:**

OW-DETR builds on previous works code base such as [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [Detreg](https://github.com/amirbar/DETReg), and [OWOD](https://github.com/JosephKJ/OWOD). If you found OW-DETR useful please consider citing these works as well.

