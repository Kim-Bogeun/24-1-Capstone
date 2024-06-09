# Image-Text Multi-modal Sponsored Review Detection
##  24-1-Capstone AI Project

## Abstract

## Dataset
Detailed information about the dataset is in the [Data](https://github.com/Kim-Bogeun/24-1-Capstone/tree/main/Data) folder.

## Method


## Experiment
### Model Performance Comparison

| Model         | Label        | Accuracy | Precision | Recall | F1-Score |
|---------------|--------------|----------|-----------|--------|----------|
| **image (VGG19)** | Sponsored    | 0.78     | 0.81      | 0.78   | 0.80     |
|               | Unsponsored  | 0.75     | 0.79      | 0.77   | 0.77     |
| **text (Text-CNN)** | Sponsored    | 0.88     | 0.87      | 0.92   | 0.89     |
|               | Unsponsored  | 0.89     | 0.85      | 0.84   | 0.86     |
| **VSCNN**         | Sponsored    | 0.91     | 0.90      | 0.94   | 0.92     |
|               | Unsponsored  | 0.92     | 0.87      | 0.90   | 0.90     |
| **SMPC**          | Sponsored    | 0.88     | 0.92      | 0.92   | 0.90     |
|               | Unsponsored  | 0.89     | 0.98      | 0.87   | 0.97     |
| **ours**          | Sponsored    | 0.934    | 0.94      | 0.94   | 0.93     |
|               | Unsponsored  | 0.92     | 0.90      | 0.93   | 0.92     |





#### ※ Paper that was helpful for model construction and implementation of the framework.
[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)     
Github link : https://github.com/yaqingwang/EANN-KDD18

