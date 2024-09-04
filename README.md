# Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Language 

This repository contains the PyTorch implementation of the following paper:
> **Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Language**<be>
><br>
> Jeong Hun Yeo, Chae Won Kim, Hyunjun Kim, Hyeongseop Rha, Seunghee Han, Wen-Huang Cheng, Yong Man Ro<br>
> \[[Paper Link](https://arxiv.org/abs/2409.00986)\]


## Introduction
We propose a novel speaker-adaptive lip reading method that adapts a pre-trained lip reading model to target speakers at both vision and language levels. Specifically, we integrate prompt tuning and the LoRA approach, applying them to a pre-trained lip reading model to effectively adapt the model to target speakers. In addition, to validate its effectiveness in real-world scenarios, we introduce a new dataset, VoxLRS-SA, derived from VoxCeleb2 and LRS3.
<div align="center"><img width="80%" src="img/img.png?raw=true" /></div>




## Environment Setup
```bash
conda create -n Personalized-Lip-Reading python=3.9 -y
conda activate Personalized-Lip-Reading
git clone https://github.com/JeongHun0716/Personalized-Lip-Reading
cd Personalized-Lip-Reading
```
```bash
# PyTorch and related packages
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
(If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode librosa
pip install omegaconf==2.0.6 hydra-core==1.0.7
pip install transformers peft bitsandbytes
cd fairseq
pip install --editable ./
```


## Dataset preparation
To validate speaker adaptive lip reading methods in real-world scenarios, we proposed a new dataset named VoxLRS-SA.
To download the VoxLRS-SA dataset, please refer to [this repository](https://github.com/JeongHun0716/VoxLRS-SA/tree/main).


## Training and Inference
1. Baseline lip-reading model

To train the baseline lip-reading model, please download the checkpoints from the links below and move them to the target directory.

| Conformer Encoder Model    | Training Datasets    | Target Directory |
|--------------|:----------|:------------------:|
| [vsr_trlrs3_base.pth](https://github.com/mpc001/auto_avsr) |     LRS3   |  src/pretrained_models/conformer_encoder/pretrained_lrs3   |


| Large Language Model     | Target Directory |
|--------------|:----------|
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)   |  src/pretrained_models/llm   |

```bash
# Train
bash scripts/train/baseline/train.sh
# Evaluation
bash scripts/eval/baseline/eval.sh
```
The pre-trained baseline lip-reading model is provided in the below link.

|  Baseline Model      | Training Datasets  |  WER(\%)  | Target Directory |
|--------------|:----------|:------------------:|:----------:|
| [best_ckpt.pt](https://www.dropbox.com/scl/fi/x6c35ubgnzi08v6pomynj/checkpoint_best.pt?rlkey=9zzb54lx8b8xac04kqxw947if&st=nxf1cx7p&dl=0) |     VoxLRS-SA             |     47.3  |  src/pretrained_models/conformer_encoder/pretrained_w_llm   |

2. Vision Level Adaptation to Target speaker
```bash
# Train
bash scripts/train/adaptation/vision_level/train.sh
# Evaluation
bash scripts/eval/adaptation/vision_level/eval.sh
```
The pre-trained vision-adapted model is provided in the below link.

| Vision Adapted Model      | Training Datasets  |  WER(\%)  | Target Directory |
|--------------|:----------|:------------------:|:----------:|
| [best_ckpts.zip](https://www.dropbox.com/scl/fi/s1c1sjaai7ff2i4vvu6bd/vision.zip?rlkey=fni3jv61kaoya4zwtyr3lu39y&st=9pbz0fxg&dl=0) |     VoxLRS-SA              |     41.5  |  src/pretrained_models/adapted_model/vision   |

3. Vision & Language Levels Adaptation to Target speaker
```bash
# Train
bash scripts/train/adaptation/vision_language_level/train.sh
# Evaluation
bash scripts/eval/adaptation/vision_language_level/eval.sh
```
The pre-trained vision- and language-adapted model is provided in the below link.

| Vision \& Language Adapted Model     | Training Datasets  |  WER(\%)  | Target Directory |
|--------------|:----------|:------------------:|:----------:|
| [best_ckpts.zip](https://www.dropbox.com/scl/fi/wxrqy28h0xlevak09pqwj/vision_language.zip?rlkey=epxy0h2exwrcnv4fctsledmy9&st=zfblhfov&dl=0) |      VoxLRS-SA            |     40.9  |  src/pretrained_models/adapted_model/vision_language   |


The adapted pre-trained models should be unzipped in the Target Directory, to evaluate the performance in the VoxLRS-SA dataset.


## Acknowledgement
This project is based on the [avhubert](https://github.com/facebookresearch/av_hubert), [espnet](https://github.com/espnet/espnet), and [fairseq](https://github.com/facebookresearch/fairseq) code. We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.
