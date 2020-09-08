# Semi-Supervised Learning on MNIST Dataset

## Table of Content

1. [Introduction](#Introduction)
2. [Data](#Data)
3. [Method](#Method)
4. [Files](#Files)
5. [Results](#Results)
6. [Conclusion](#Conclusion)

## Introduction

In this challenge, I performed a semi-supervised learning on MNIST dataset. Semi-supervised learning approach is very useful when there is small amount of labeled data and large amount of unlabeled data. There are several powerful methods in this field. 

## Data

MNIST dataset contains images of handwritten digits and the images are 28x28 pixels which makes 784 when we have the images as vectors. In this challenge, we have access to whole MNIST dataset, but we are only allowed to use 10 labels (1 per each class). Therefore, rest of the data should be unlabeled. For the test data we use the original test split of MNIST dataset.  

Dataset size details are:  
- Labeled Data   : 10  
- Unlabaled Data : 59998
- Test Data      : 10000

## Method

I built AutoEncoder models to learn a latent representation of the image data. This means that the model is trained to produce numerical features that can be used for another classification or other tasks later on. The training is done by using images as input and passing through Encoder and Decoder components of the model. Encoder basically aims to downscale the input to a latent feature space and after this operation, Decoder aims to upscale and reconstruct the input image from these features. In this way, the model learns 2 tasks, extracting meaningful features that it can be used for reconstructing the image. There are different types of Loss metrics that can be used for this reconstruction task. As this is a toy example, I decided to use Mean Squared Error (MSE) which works pixel-wised.

I implemented two different architecture for AutoEncoder: (1) Linear AutoEncoder which contains Linear (Fully Connected) layers, and (2) Convolutional AutoEncoder which contains Convolutional Layers. By evaluating 2 different type of layer, we will see which one performs better on this small sized image data.

After extracting representations by using trained Autoencoders, I followed a simple approach and used K-Nearest Neighbour algorithm. I made experiments by using different representations. Firstly, I used directly labaled (10) images as vector. Secondly, I used extracted features which are 128 float numbers from implemented autoencoder types and only labaled data again. Finally, I labeled the unlabeled data with best performing features with KNN model and fit KNN model again with new training dataset. The results of these experiments are listed in Results section.


## Files

* /
    * **data**
        * `mnist.pkl`  
            This file will be downloaded after running downloader script.
    * **models**
        * `AutoEncoder_Convolutional.py`  
            Convolutional AutoEncoder implementation.
        * `AutoEncoder_Linear.py`  
            Linear AutoEncoder implementation.
    * **util**  
        * `mnist_loader.py`  
            Helper functions to load mnist data without pain.
        * `mnist-downloader.py`  
            Helper functions that downloads MNIST dataset and saves it as pickle file to **data** directory
        * `ModelHandler.py`  
            Class implementation for trainings of Autoencoder Models
    * `Notebook.ipynb`  
        Jupyter-Notebook that runs all of the pipeline for trainings and classifications.


## Results

We can see the classification accuracy and F1-Score in the table below. Basically, I have 2 baseline, one of them is a Dummy Random Classifier which performs 10% accuracy and second baseline is the raw image with KNN model. In second one, I do not perform any representation learning, I just want to see the performance with 10 labels and images.  
In the second step, I trained two autoencoder models to extract meaningful features which are sized 168 float numbers in both approaches. We can see that Linear AutoEncoder could not even pass Raw image data model. However, it is clear to see that Convolutional AutoEncoder achieved to extract better features than Linear version, and it outperformed by 9% of Accuracy and F1-Score.  
As a last classifier, I labeled the unlabeled data by using the KNN model with Convolutional AutoEncoder and 10 labaled data. After that with this new dataset, I fit another KNN model and we got 65% Accuracy which is 9% better than using only 10 labels with same extracted features.


| Algorithm                 | Accuracy     | F1-Score |
|---------------------------|--------------|----------|
| Dummy Classifier (Random) | 10           | -        |
| KNN (Raw Labaled Data)    | 49           | 48       |
| KNN (Linear AutoEncoder)  | 45           | 44       |
| KNN (Conv AutoEncoder)    | 56           | 55       |
| KNN (Conv AutoEncoder) *  | **65**       | 64       |

(*) Unlabaled data is classified first, and new dataset is created to train KNN.

## Conclusion

We can see that, with feature extraction we could get some reasonable classifiers. However, this is a very simple dataset and we should have performed better. The biggest problem is having only 10 labels when it comes to classification. Better performance could be achieved by having a strategy to choose one the most representable image for each class with some Unsupervised Learning like KMeans or statistical approach, and use them to label other unlabeled images with the same method I have done. However, I did not want to choose images and I assumed that the labels that I have are predefined and it is impossible to label manually. Therefore, I picked labeled data randomly from dataset. 