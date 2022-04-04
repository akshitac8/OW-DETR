# OW-DETR: Open-world Detection Transformer (CVPR 2022)

[[`Paper`](https://arxiv.org/pdf/2112.01513.pdf)]

#### [Akshita Gupta](https://akshitac8.github.io/)<sup>\*</sup>, [Sanath Narayan](https://sites.google.com/view/sanath-narayan)<sup>\*</sup>, [K J Joseph](https://josephkj.in), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) ####

(:star2: denotes equal contribution)

## Introduction

 Open-world object detection (OWOD) is a challenging computer vision problem, where the task is to detect a known set of object categories while simultaneously identifying unknown objects. Additionally, the model must incrementally learn new classes that become known in the next training episodes. Distinct from standard object detection, the OWOD setting poses significant challenges for generating quality candidate proposals on potentially unknown objects, separating the unknown objects from the background and detecting diverse unknown objects. Here, we introduce a novel end-to-end transformer-based  framework, OW-DETR, for open-world object detection. The proposed OW-DETR comprises three dedicated components namely, attention-driven pseudo-labeling, novelty classification and objectness scoring  to explicitly address the aforementioned OWOD challenges. Our OW-DETR explicitly encodes multi-scale contextual information, possesses less inductive bias, enables knowledge transfer from known classes to the unknown class and can better discriminate between unknown objects and background. Comprehensive experiments are performed on two benchmarks: MS-COCO and PASCAL VOC. The extensive ablations reveal the merits of our proposed contributions. Further, our model outperforms the recently introduced OWOD approach, ORE, with absolute gains ranging from  $1.8\%$ to $3.3\%$ in terms of unknown recall on MS-COCO. In the case of incremental object detection, OW-DETR outperforms the state-of-the-art for all settings on PASCAL VOC.
 
<p align="center" ><img width='350' src = "https://imgur.com/KXDXiAB.png"></p> 

<br>

<p align="center" ><img width='500' src = "https://imgur.com/cyeMXuh.png"></p>

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n owdetr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate owdetr
    ```
    Installation: (change cudatoolkit to your cuda version. For detailed pytorch installation instructions click [here](https://pytorch.org/))
    ```bash
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


## Dataset preparation

### [OWOD](https://github.com/JosephKJ/OWOD) paper splits 

<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

The splits are present inside `data/VOC2007/OWOD/ImageSets/` folder. The remaining dataset using this [link](https://drive.google.com/drive/folders/11bJRdZqdtzIxBDkxrx2Jc3AhirqkO0YV)

The files should be organized in the following structure:
```
code_root/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```

### New proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

The splits are present inside `data/VOC2007/OWDETR/ImageSets/` folder.

1. Please download [COCO 2017 dataset](https://cocodataset.org/) inside `data/` folder.
2. Transfer images from train2017 and val2017 folders to `data/VOC2007/OWDETR/JPEGImages/`.
3. Run `coco2voc.py` to convert all coco annotations to VOC format and add them to `data/VOC2007/OWDETR/Annotations/`.

All the above can be skipped if coco dataloader is followed. (Update coming soon..)

The files should be organized in the following structure:
```
code_root/
└── data/
    └── VOC2007/
        └── OWDETR/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```


Currently, Dataloader followed for OW-DETR is in VOC format.

## Training

#### Training on single node

Command for training OW-DETR which is based on Deformable DETR on 8 GPUs is as following:

```bash
./run.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 2 node with 8 GPUs each:

```bash
sbatch run_slurm.sh
```

## Evaluation

You can get the config file and pretrained model of OW-DETR (the link is in "Results" session), then run following command to evaluate it on test set:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

*Note:*
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.


## Results

### Reported results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=2>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center"></td>
    </tr>
</table>


### Reproduced results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=3>Task1</th>
        <th align="center" colspan=3>Task2</th>
        <th align="center" colspan=3>Task3</th>
        <th align="center" colspan=3>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
    </tr>
</table>

### Improved reproduced results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=3>Task1</th>
        <th align="center" colspan=3>Task2</th>
        <th align="center" colspan=3>Task3</th>
        <th align="center" colspan=3>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
        <td align="center">URecall</td>
        <td align="center">mAP</td>
        <td align="center">URL</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
        <td align="center"></td>
        <td align="center"></td>
        <td align="center">URL</td>
    </tr>
</table>


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Citation

If you use OW-DETR, please consider citing:

    @inproceedings{gupta2021ow,
        title={OW-DETR: Open-world Detection Transformer}, 
        author={Gupta, Akshita and Narayan, Sanath and Joseph, KJ and 
        Khan, Salman and Khan, Fahad Shahbaz and Shah, Mubarak},
        booktitle={CVPR},
        year={2022}
    }

## Contact

Should you have any question, please contact :e-mail: akshita.sem.iitr@gmail.com

**Acknowledgments:**

OW-DETR builds on previous works code base such as [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [Detreg](https://github.com/amirbar/DETReg), and [OWOD](https://github.com/JosephKJ/OWOD). If you found OW-DETR useful please consider citing these works as well.
