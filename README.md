# Self-Research 1 - Implementation and Analysis of Deep Learning Models
This repository contains the code implementation for **Self-Research 1**, a project focused on the implementation, evaluation, and optimization of deep learning models for computer vision tasks.

## Project Overview
The objective of this project is to explore and analyze the performance of various deep learning architectures for image classification. Through this research, we aim to compare and optimize the capabilities of CNN-based and Transformer-based models, focusing on:

- **ResNet**: Addressing the degradation problem in deep networks using residual connections.
- **Vision Transformer (ViT)**: Investigating the effectiveness of Transformer-based architectures in image classification tasks.
- **ViT Pretrained**: Evaluating pretrained Vision Transformer models using the `torchvision` library.
- **Further Experiments**: Exploring advanced techniques like hyperparameter tuning and augmentation methods (e.g., CutMix).

All models were tested on dataset **CIFAR-100** to evaluate metrics like accuracy and error rates.

## Directory Structure
The following is the directory structure of this repository:

```yaml
selfResearch/
├── models/
│   ├── resnet.py          # ResNet model implementation
│   ├── vgg.py             # VGGNet model implementation
│   ├── vit_pretrained.py  # Pretrained Vision Transformer implementation
│   ├── vit_scratch.py     # Vision Transformer trained from scratch
├── data_transform.py      # Data augmentation and preprocessing utilities
├── main.py                # Main script for training and evaluation
├── utils.py               # Helper functions and utilities
├── requirements.txt       # List of dependencies required to run the project
└── README.md              # Project documentation
```  

## Experiment Results  
### Overview
The experiments were conducted to evaluate the performance of different deep learning models (e.g., CNN and Vision Transformers) on image classification tasks.

### Dataset
The models were tested on the CIFAR-100 dataset, which consists of : 
- **Training Images** : 50,000
- **Testing Images** : 10,000
- **Classes** : 100

Each image is a 32x32 colored image, suitable for benchmarking deep learning models.

### Evaluation Metrics
- **Top-1 Accuracy** : The percentage of predictions where the top predicted class matches the ground truth.
- **Top-5 Accuracy** : The percentage of predictions where the ground truth is within the top 5 predicted classes.

### Results Summary
```markdown
| Model                    | Top-1 Accuracy (%)| Top-5 Accuracy (%) |
|--------------------------|-------------------|--------------------|
| VGGNet                   | 75.4              | 93.38              |
|--------------------------|-------------------|--------------------|
| ResNet-50                | 79.2              | 94.52              |
|--------------------------|-------------------|--------------------|
| Vision Transformer (ViT) | 85.6              | 98.4               |
|--------------------------|-------------------|--------------------|
| ResNet (Tuned)           | 80.56             | 95.04              |
|--------------------------|-------------------|--------------------|
```

## How To Use
### 1. Clone the Repository
clone the Repository and navigate to the project directory : 
```bash
git clone https://gitgub.com/meansash/self_research_1.git
cd self_research_1
```
### 2. Install Dependencies
Install the required dependencies listed in the `requirements.txt` file : 
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Ensure the dataset (e.g., CIFAR-100, ImageNet1K) is downloaded and specify the dataset path using the `--datapath` argument.

### 4. Run the Training Script
Use the following commands to train & evaluate specific models. Replace `<path-to-dataset>` with the actual path to your dataset.
#### Train VGG
```bash
python main.py --gpu 0 --model vgg16 --lr 0.01 --batch_size 128 --epochs 200 --augmentation cutmix --optimizer sgd --weight_decay 0.0002 --loss CE --datapath <path-to-dataset> --wandb_run_name vgg16
```
#### Train ResNet
```bash
python main.py --gpu 0 --model resnet50 --lr 0.01 --batch_size 128 --epochs 200 --augmentation cutmix --optimizer sgd --weight_decay 0.0002 --loss CE --datapath <path-to-dataset> --wandb_run_name rn50_sgd_CE
```
#### Train ResNet (Hyper parameter tuning according to paper `ResNet Strikes Back : An improved training procedure in timm`)
```bash
python main.py --gpu 0 --model resnet50 --lr 0.01 --batch_size 128 --epochs 200 --augmentation cutmix --optimizer adamw --weight_decay 0.02 --loss BCE --warmup_epochs 5 --datapath <path-to-dataset> --wandb_run_name rn50_adamw_BCE
```
#### Train Vision Transformer (Pretrained model Fine-tuning) 
```bash
python main.py --gpu 0 --model vit_b_16 --lr 0.01 --datapath <path-to-dataset> --batch_size 128 --epochs 50 --augmentation cutmix --wandb_run_name vit_b_16
```
#### Train Vision Transformer (From Scratch)
```bash
python main.py --gpu 0 --model vit_scratch --lr 0.01 --batch_size 64 --epochs 100 --optimizer sgd --weight_decay 0.0002 --datapath <path-to-dataset> --wandb_run_name vit_from_scratch
```
## Future Work
- Experiment with additional datasets like ImageNet and Tiny ImageNet.
- Explore Vision-Language Models (e.g., CLIP) for multimodal learning.
- Investigate and research Medical AI models utilizing Vision-Language Models (VLM), focusing on applications such as medical image analysis and diagnosis