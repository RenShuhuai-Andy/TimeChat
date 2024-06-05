<p align="center" width="100%">
<a target="_blank"><img src="figs/logo.png" alt="TimeChat" style="width: 40%; min-width: 150px; display: block; margin: auto;"></a>
</p>

<h2 align="center"> <a href="https://arxiv.org/abs/2312.02051">TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding</a></h2>

<h4 align="center"> <a href="https://renshuhuai-andy.github.io/">Shuhuai Ren*</a>, <a href="https://yaolinli.github.io/">Linli Yao*</a>, <a href="https://lscpku.github.io/">Shicheng Li</a>, <a href="https://xusun26.github.io/">Xu Sun</a>, <a href="https://houlu369.github.io/">Lu Hou</a></h4>

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2312.02051'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='https://huggingface.co/datasets/ShuhuaiRen/TimeIT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> 
<a href='https://huggingface.co/ShuhuaiRen/TimeChat-7b'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a> 
</div>

## News
- [24.06.04] ![NEW!](https://img.shields.io/badge/NEW!-red) Add FAQ, see [docs](./docs/FAQ.md).
- [24.06.04] ![NEW!](https://img.shields.io/badge/NEW!-red) Release evaluation results of TimeChat-7b on several VideoLLM benchmarks (e.g., MVBench, TempCompass, etc.), see [docs](./docs/EVAL.md).
- [24.01.09] Release **TimeChat-7b** 🤗 [checkpoint](https://huggingface.co/ShuhuaiRen/TimeChat-7b) and [local demo](./demo.ipynb).
- [23.12.27] 🤗 Release the instruction-tuning dataset of **[TimeIT](https://huggingface.co/datasets/ShuhuaiRen/TimeIT)**.
- [23.12.06] Release the initial version of **TimeChat**. 

<p align="center" width="100%">
<a target="_blank"><img src="figs/arch.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Introduction

- **TimeChat** is a time-sensitive multimodal large language model specifically designed for long video understanding. Our model incorporates two key architectural contributions: 
  - (1) a timestamp-aware frame encoder that binds visual content with the timestamp of each frame
  - (2) a sliding video Q-Former that produces a video token sequence of varying lengths to accommodate videos of various durations. 
- We also construct an instruction-tuning dataset named **TimeIT**, encompassing 6 tasks and a total of 125K instances, to further enhance TimeChat's instruction-following performance.


## Example Outputs
- **An illustration of temporal localization capability of TimeChat**

<p float="left">
    <img src="figs/teaser.png" style="width: 100%; margin: auto;">
</p>

- **Examples for dense video captioning (left), temporal video grounding (middle), and video highlight detection (right)**

<p float="left">
    <img src="figs/case_dvc.png" style="width: 32%; margin: auto;">
    <img src="figs/case_tvg.png" style="width: 34%; margin: auto;">
    <img src="figs/case_vhd.png" style="width: 32%; margin: auto;">
</p>



## Fine-tuned Checkpoints

The following checkpoints store learnable parameters (positional embedding layers, Time-aware Frame Encoder, Sliding Video Q-Former, linear projection layers, and lora) only.

| Checkpoint              | LLM backbone | Link                                                                            | Note                                                                                                                                                                                                                                                                                                                                                         |
|:------------------------|--------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TimeChat-2-7B-Finetuned | LLaMA-2 7B   | [link](https://huggingface.co/ShuhuaiRen/TimeChat-7b/blob/main/timechat_7b.pth) | Fine-tuned on the instruction-tuning data from [TimeIT-104K](https://huggingface.co/datasets/ShuhuaiRen/TimeIT) (asr version) and [Valley-73K](https://huggingface.co/datasets/ShuhuaiRen/TimeIT/blob/main/data/valley/Valley_instruct_73k.json) (previous version of current [Valley-65K]((https://huggingface.co/datasets/luoruipu1/Valley-Instruct-65k))) |


## Usage
#### Enviroment Preparation 

First, install ffmpeg.
```
apt update
apt install ffmpeg
```
Then, create a conda environment:
```
conda env create -f environment.yml
conda activate timechat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```


## Prerequisites

Before fine-tuning your own model (or reproduce our TimeChat model), make sure you have obtained the following checkpoints:

#### Pre-trained Image Encoder (EVA ViT-g)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```

#### Pre-trained Image Q-Former (InstructBLIP Q-Former)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
```

#### Pre-trained Language Decoder (LLaMA-2-7B) and Video Encoder (Video Q-Former of Video-LLaMA)

Use `git-lfs` to download weights of [Video-LLaMA (7B)](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main):
```bash
git lfs install
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned
```

#### Instruct-tuned [TimeChat-7B](https://huggingface.co/ShuhuaiRen/TimeChat-7b)
```bash
git lfs install
git clone https://huggingface.co/ShuhuaiRen/TimeChat-7b
```

The file structure looks like:
```
ckpt/
|–– Video-LLaMA-2-7B-Finetuned/
    |-- llama-2-7b-chat-hf/
    |-- VL_LLaMA_2_7B_Finetuned.pth
|–– instruct-blip/
    |-- instruct_blip_vicuna7b_trimmed.pth
|–– eva-vit-g/
    |-- eva_vit_g.pth
|-- timechat/
    |-- timechat_7b.pth
```

## How to Run Demo Locally

Please refer to our Jupyter Demo [here](./demo.ipynb).

## Instruction-Tuning

### Data
For now, the fine-tuning dataset consists of:
* 104K time-sensitive instructions from **TimeIT** [[link](https://huggingface.co/datasets/ShuhuaiRen/TimeIT)]
  * see [DATA.md](./docs/DATA.md)
* 73K (now 65K) video-based instructions from **Valley** [[link](https://huggingface.co/datasets/luoruipu1/Valley-Instruct-65k)]

### Script

#### Tuning
Config the checkpoint and dataset paths in [stage2_finetune_time104k_valley72k.yaml](./train_configs/stage2_finetune_time104k_valley72k.yaml).
```
conda activate timechat
torchrun --nproc_per_node=8 train.py --cfg-path  train_configs/stage2_finetune_time104k_valley72k.yaml
```

#### Evaluation
Config the checkpoint and dataset paths in [timechat.yaml](./eval_configs/timechat.yaml).

Config the downstream task in [eval.sh](eval.sh).
```
bash eval.sh
```

## Recommended GPUs
* Instruction-tuning: 8xV100 (32G)
* Inference: 1xA100 (40G/80G) or 1xA6000

## Acknowledgement
We are grateful for the following awesome projects our TimeChat arising from:
* [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA): An Instruction-tuned Audio-Visual Language Model for Video Understanding
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-language Understanding with Advanced Large Language Models
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models 
* [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP): Improved Training Techniques for CLIP at Scale
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [VideoChat](https://github.com/OpenGVLab/Ask-Anything): Chat-Centric Video Understanding
* [TESTA](https://github.com/RenShuhuai-Andy/TESTA): Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding


## Term of Use
Our TimeChat is just a research preview intended for non-commercial use only. You must **NOT** use our TimeChat for any illegal, harmful, violent, racist, or sexual purposes. You are strictly prohibited from engaging in any activity that will potentially violate these guidelines. 

## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{Ren2023TimeChat,
  title={TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding},
  author={Shuhuai Ren and Linli Yao and Shicheng Li and Xu Sun and Lu Hou},
  journal={ArXiv},
  year={2023},
  volume={abs/2312.02051},
}
```

