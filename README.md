# Image-Text Multi-modal Sponsored Review Detection
##  24-1-Capstone AI Project

## Abstract
This study aims to develop a multimodal approach to detect sponsored reviews on the e-commerce platform Coupang, based solely on review content without using user information. Using a balanced dataset of 6,238 reviews (3,119 sponsored and 3,119 unsponsored) collected across various product categories, features were extracted from text, images, and metadata. The proposed model employs an intermediate fusion approach, combining text features extracted with a Text-CNN model, visual features from a pre-trained VGG19 network, and metadata features through a fully connected layer. This model outperformed baseline approaches like VSCNN and SMPC, achieving an overall accuracy of 0.934, precision of 0.94/0.92, and recall of 0.93/0.93 for sponsored/unsponsored reviews respectively. Analysis revealed that sponsored reviews tend to have more images, higher ratings, longer text, and more structured formatting (e.g., more line breaks, presence of titles) compared to unsponsored reviews. The study demonstrates the feasibility of identifying sponsored reviews based solely on content, which has academic implications for advancing content-based detection methods and practical implications for maintaining consumer trust and brand integrity. Limitations include the inability to use multiple images per review and the need for a larger dataset, suggesting areas for future work.

## Dataset
Detailed information about the dataset is in the [Data](https://github.com/Kim-Bogeun/24-1-Capstone/tree/main/Data) folder.

## Model Structure
![image](https://github.com/Kim-Bogeun/24-1-Capstone/assets/127417159/70c1ab9a-d850-43cd-a7d4-2158f6870776)

## Experiment
### Model Performance Comparison
![image](https://github.com/Kim-Bogeun/24-1-Capstone/assets/127417159/1a22a4d6-b672-477f-813f-b3113d309e85)




   
#### ※ Paper that was helpful for model construction and implementation of the framework.
[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)     
Github link : https://github.com/yaqingwang/EANN-KDD18

