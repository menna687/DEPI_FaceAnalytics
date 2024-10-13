# Face Recognition & Analytics
"Multitask Face Recognition with Age and Gender Detection using Deep Learning"   
This project aims to build a multi-task learning model that can simultaneously perform face recognition, age estimation, and gender classification.

## Project Outline

## Dataset
Using selected features from [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
CelebA dataset consists of triplets of images (anchor, positive, negative) used for faace recognition in addition to analytic features describing each image such as: gender, hair color, glasses, age, ...

### Data preprocessing
In the data preprocessing step, images are resized, matched to their triplets, and each anchor image is matched to its features that will be used in training which are age and gender.

## Model Architecture
The model consists of:
* Backbone convolutional neural network (CNN) used to extract images features (anchor, positive and negative). InceptionV3 pretrained model is used.
* Feed forward network head to detect anchor image age
* Feed forward network head to detect anchor image gender
* Cosine similarity head to detect similarity between anchor image and other images

### Model Training 
The backbone CNN weights are freezed except for the last few layers. The other heads are all tainable.

#### Loss function:
* For face recognition task -> cosine simillarity
* For gender and age -> Binary cross entropy loss

## Model Deployment
