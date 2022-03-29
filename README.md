 # CSRNet Model
 Yuhong Li et al, propose a successful solution using a deep CNN, called CSRNet, network for crowded scene estimation. CSRNet generates complexity-free density maps. The network included expanded receptive fields that extract high-level information due to the dilated convolutional layers used. In comparison with the previous state-of-the-art models, CSRNet succeed upon achieving the best accuracy (the performance was improved by 47.3% lower Mean Absolute Error (MAE) in some cases). The distinctive work of was implemented as the core method of Fawj app. In this section we will present network architecture, training details, and results of the implemented method. 
 

# Dataset :
The experiments were accompanied on ShanghaiTech dataset Part A , which has different density levels and also has different complex scenes the ShanghaiTech dataset consists of 1198 images and 330,165 heads. Part A includes 482 images randomly selected from the Internet. For
experiments, Part A was divided into training and test sets. 300 images in Part A were used for training and 182 in testing: [Kaggle Link](https://www.kaggle.com/tthien/shanghaitech)

The dataset is divided into two parts, A and B. Part A consists of images with a high density of crowd. Part B consists of images with sparse crowd scenes.   

# Data Preprocessing  :
In `Preprocess.ipynb` notebook, the main objective was to convert the ground truth provided by the ShanghaiTech dataset into density maps.


# Model
The model consists of a front-end and a back-end. The first ten layers of VGG-16, a CNN proposed by K. Simonyan and A. Zisserman, as well as three pooling layers used as the front end while the fully-connected layers omitted to reduce the overhead. The convolutional layers use a kernel of size 3 × 3 and three pooling layers with a stride of 2. The size of the density maps generated is 1/8 of the original input size. The model uses six dilated convolutional layers as the back-end with a kernel of size 3 × 3 and dilation rate of 2 to extract deeper features and generate high-quality density maps also to preserve the resolution of the output. All hidden layers use ReLU activation function, except the output layer which applies a softmax. See the `Model.ipynb` notebook.


# Inference :

The experiments were confirmed on ShanghaiTech Part A dataset. The model is straightforward and uncomplicated, trained and evaluated on a Mac OS Intel(R) Core(TM) i5-5257U CPU 2.70GHz using TensorFlow framework as backend keras. Because of the limitation of processor speed and the graphics card, the training required extra computational time. Refer to the `Inference.ipynb` for generating inference. 

Given below is the result on actual images taken from the test set provided in the ShanghaiTech dataset.

Actual Image :

<img src="https://github.com/Arwa-Alshanbari/Fowj/main/test_images/IMG_105.jpg" width="480">

Generated Density Map : 

<img src="https://github.com/Arwa-Alshanbari/Fowj/main/results/105.jpg" width="480">

Actual Count : 258

Predicted Count : 232

# Result :

The implemented model performs better with MAE of 66.4 and MSE of 92.4 Compared to original CSRNet model with MAE of 68.2 and MSE of 115.0.

|       Dataset       | MAE           |  MSE
| ------------------- | ------------- |---------------|
|CSRNet               | 68.2          |115            |
|Fawj                 |66.4           |92.4           |


# Requirements :

1. Keras 
2. Tensorflow 
3. Scipy 
4. Numpy
5. Pillow(PIL)
6. OpenCV

# References 
Paper: [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes.](https://arxiv.org/abs/1802.10062).
Official Pytorch Implementation( https://github.com/Neerajj9/CSRNet-keras).
