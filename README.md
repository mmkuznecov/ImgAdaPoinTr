# ImgAdaPoinTr: Improving Point Cloud Completion via Images and Segmentation

This repository contains implementation for "ImgAdaPoinTr: Improving Point Cloud Completion via Images and Segmentation".

![scheme](fig/imgadapointr_scheme.png)

## Usage

### Installation

In order to install all requirements, run following command in terminal:

```bash
pip install -r requirements.txt
```

In order to build all necessary extensions, run:

```bash
bash install.sh
```

You can also build them manually by running

```bash
python setup.py build && python setup.py install
```

In corresponding directory.

### Dataset

ImgPCN dataset can be accessed via [link](https://n-ws-z5wzc-pd11.s3pd11.sbercloud.ru/b-ws-z5wzc-pd11-irp/datasets/ImgPCN/ImgPCN.tar.gz?AWSAccessKeyId=u-ws-z5wzc-pd11-yjo&Signature=udIvocnqf2EYsWFNqa8%2FXcdpS3g%3D&Expires=1698403154).


### Run

To run train:

```bash
bash ./scripts/train.sh 0 --config ./cfgs/ImgPCN_models/ImgResNetEncAdaPoinTrVariableLoss.yaml  --exp_name train_ImgResNetEncAdaPoinTrVariableLoss --num_workers 16 --val_freq 1
```

To eval:

```bash
bash ./scripts/test.sh 0 --ckpts experiments/ImgResNetEncAdaPoinTrVariableLoss/ImgPCN_models/train_ImgResNetEncAdaPoinTrVariableLoss_easy/ckpt-best.pth --config ./cfgs/ImgPCN_models/ImgResNetEncAdaPoinTrVariableLoss.yaml --exp_name test
```