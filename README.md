# Image-Text Multi-modal Sponsored Review Detection
##  24-1-Capstone AI Project

## Abstract
This study aims to develop a multimodal approach to detect sponsored reviews on the e-commerce platform Coupang, based solely on review content without using user information. Using a balanced dataset of 6,238 reviews (3,119 sponsored and 3,119 unsponsored) collected across various product categories, features were extracted from text, images, and metadata. The proposed model employs an intermediate fusion approach, combining text features extracted with a Text-CNN model, visual features from a pre-trained VGG19 network, and metadata features through a fully connected layer. This model outperformed baseline approaches like VSCNN and SMPC, achieving an overall accuracy of 0.934, precision of 0.94/0.92, and recall of 0.93/0.93 for sponsored/unsponsored reviews respectively. Analysis revealed that sponsored reviews tend to have more images, higher ratings, longer text, and more structured formatting (e.g., more line breaks, presence of titles) compared to unsponsored reviews. The study demonstrates the feasibility of identifying sponsored reviews based solely on content, which has academic implications for advancing content-based detection methods and practical implications for maintaining consumer trust and brand integrity. Limitations include the inability to use multiple images per review and the need for a larger dataset, suggesting areas for future work.

## Dataset
Detailed information about the dataset is in the [Data](https://github.com/Kim-Bogeun/24-1-Capstone/tree/main/Data) folder.

## Model(Method)
![image](https://github.com/Kim-Bogeun/24-1-Capstone/assets/127417159/70c1ab9a-d850-43cd-a7d4-2158f6870776)
An overview of our model is presented in Fig2. We adopted intermediate fusion for this task. This approach was chosen because it showed better performance compared to early fusion and late fusion methods. Our model consists of modules that extract features from different modalities and a part that integrates these features.

**a) Text Feature Module**

The input to this module is the list of words of the reviews. we employ convolutional neural networks(Text-CNN) to extract textual feature, similar to the approach used by Wang et al. in EANN. (인용?) 

First, each word in the text is represented as a word embedding vector. For the $i$-th word in the sentence, the corresponding $k$-dimensional word embedding vector is denoted as $T_i \in R^k$. Thus, a sentence with n words is represented as:

$$
\mathbf{T}_{1:n} = \mathbf{T}_1 \oplus \mathbf{T}_2 \oplus \ldots \oplus \mathbf{T}_n
$$

where $\oplus$ is the concatenation operator. In the convolution layer, multiple filters of varying sizes are applied to capture different n-gram features in the text. Specifically, the operation of convolutional filter with a window size $h$ starting with the $i$-th word can be represented as:

$$
t_i = \sigma(W_c \cdot T_{i:i+h-1})
$$

where $\sigma(·)$ is the ReLU activation function and $W_c$ is the weight of the filter. Then we get a feature vector $t = [t_1, t_2, \cdots ,t_{n-h+1}]$ for a sentence.

After the convolutional steps, we apply a max-pooling operation for each feature vector $t$ to capture the most important features and reduce the dimension. The results from the pooling operations are then concatenated into a single feature vector. Finally, the concatenated vector is passed through a fully connected layer with a LeakyReLU activation function, and $R_T$ is obtained as:

$$
R_T = \sigma(W_{tf} \cdot R^\ast_T)
$$

where ${R^\ast_T}$ is the textual features after the max-pooling, $W_{tf}$ is the weight matrix of the fully connected layer.

**b) Visual Feature Module**
The images are passed through a VGG-19 network that has been pre-trained on a ImageNet dataset. To match the required input dimension, images are resized to $224 \times 224$. The extracted feature vectors are then passed through a fully connected layer, referred to as Vis-fc. The visual feature representation $R_V$ is obtained as:

$$
R_V = \sigma(W_{vf} \cdot R^\ast_{V})
$$

where ${R^\ast_V}$ is the visual feature obtained from pre-trained VGG-19 network, and $W_{vf}$ is the weight matrix of the Vis-fc.

**c) Meta Feature Module**

This module is designed for additional features of the review data. we performed standard scaling to normalize these features, which helps stabilizing the learning process and improving the performance. The normalized vectors are then passed through a fully connected layer. With this procedure, the output of this module is denoted as:

$$
R_M = FC(M_{scaled})
$$

where $M_{scaled}$ is the normalized feature and $FC(\cdot)$ is the sequence of fully connected layer, including batch normalization, ReLU activation function, and dropout.

**d) Model Integration**

The outputs from each module (128-dimensional vectors from both text and visual modules, and a 64-dimensional vector from the meta feature module) are concatenated to a single combined feature vector denoted as $R_C=R_T \oplus R_V  \oplus R_M$. This combined vector is passed through a fully connected layer with softmax to predict whether the review is sponsored or unsponsored.

To train the model, we employ the cross-entropy loss function, which is widely used for binary classification tasks. The cross-entropy loss $L_d$ for the sponsored review detection is computed as:

$$
L_d = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

where $N$ is the number of samples, $y_i$ is the true label for the $i$-th sample, and $\hat{y_i}$ is the predicted probability that the $i$-th review is sponsored. Therefore, the objective is to minimize the  loss function $L_d$. Adam optimizer is used with a learning rate of 0.0005.


## Experiment
### Model Performance Comparison
![image](https://github.com/Kim-Bogeun/24-1-Capstone/assets/127417159/1a22a4d6-b672-477f-813f-b3113d309e85)




   
#### ※ Paper that was helpful for model construction and implementation of the framework.
[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)     
Github link : https://github.com/yaqingwang/EANN-KDD18

