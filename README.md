# TKD: Triple Knowledge Distillation for Industrial Defect Detection

The official implementation of `TKD: Triple Knowledge Distillation for Industrial Defect Detection`.


## Introduction

Knowledge distillation stands as a potent model compression technique in object detection, notable for its capability to transfer knowledge adeptly from intricate teacher networks to more streamlined student networks. However, the efficacy of semantic knowledge transfer encounters challenges in industrial settings. Specifically, class-specific features in industrial data exhibit weak semantic associations and strong semantic ambiguities due to the agnosticism of industrial sites. To address these challenges, we propose Triple Knowledge Distillation (TKD), focusing on comprehensive transfer of semantic knowledge.
On one front, we devise dual-relation distillation (DRD), employing graph reasoning networks to intuitively reinforce semantic associations at the instance and pixel levels. On another front, we present decoupled expert distillation (DED), heightening the semantic explicitness of class-specific features by disentangling foreground-background and injecting expert prior judiciously. Additionally, we put forth cross-response distillation (CRD), aligning task-specific distributed knowledge to further inhibit industrial agnostic attributes. Our TKD approach facilitates
the triple alignment of relations, features, and responses, mitigating the impact of uncertainty prevalent in the industry.
Experimental evaluations on two demanding industrial datasets showcase that our TKD yields substantial detection performance enhancements.


## Install

### 1. Prerequisites

**Dependencies**

- Ubuntu >= 20.04
- CUDA >= 11.3
- pytorch==1.12.1
- torchvision=0.13.1
- mmcv==2.0.0rc4
- mmengine==0.7.3

Our implementation based on MMDetection==3.0.0rc6. For more information about installation, please see the [official instructions](https://mmdetection.readthedocs.io/en/3.x/).

**Step 0.** Create Conda Environment

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 1.** Install [Pytorch](https://pytorch.org)

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Step 2.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine==0.7.3"
mim install "mmcv==2.0.0rc4"
```

**Step 3.** Install [TKD](https://github.com/ZhitaoWen/TKD.git).

```shell
git clone https://github.com/ZhitaoWen/TKD
cd TKD
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
# Dataset prepare

**Step 4.** Prepare dataset follow the [official instructions](https://mmdetection.readthedocs.io/en/3.x/user_guides/dataset_prepare.html).



### 2. Training


```shell
python tools/train.py configs/tkd/${CONFIG_FILE} [optional arguments]
```


### 3. Evaluation

```shell
python tools/test.py configs/tkd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```




## License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.


## Acknowledgement

This repo is modified from open source object detection codebase [MMDetection](https://github.com/open-mmlab/mmdetection) and [CrossKD](https://github.com/jbwang1997/CrossKD).

