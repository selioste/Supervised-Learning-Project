# Supervised-Learning-Project

### INTRODUCTION

Automatic recognition of different tissue types in histological images is an essential part of the digital pathology toolbox. Texture analysis is commonly used to solve this problem, mainly in the context of estimating the tumor/stroma ratio on histological samples. However, although histological images typically contain more than two tissue types, only a few studies have addressed the multi-class problem. For colorectal cancer for example, one of the most common tumor types, there are no published results on multi-class texture separation. 
To complete our knowledge about Machine Learning model, for this project, we will work on a dataset of 5,000 human colorectal cancer histological images including eight different tissue types using supervised machine learning model. We will start with some basic models seen in class this year. We will try to compare them to find the model that seems to us the most adapted to this study. Then, we will try to go deeper with more complex models like the convolution network and the ResNet.

### DATASET EXPLORATION

The dataset we have is a collection of textures in colorectal cancer histology. It is considered as a MNIST by biologists and can be easily found on Zenodo or on Kaggle. It contains two folders, but we will consider the folder containing the 5000 images of 150x150 pixels resolution where each image belongs exactly to one of the eight tissue categories.
In this dataset, there are exactly 8 classes with 625 images per class. It's therefore a multiclass classification. However, in addition to the images, the dataset also provides individual pixel values for each image at different resolutions. To get a better understanding of the pixel pattern for each class, we used the pixel values calculated at the resolution of 64x64 pixels, i.e., in gray scale to make statistics study. 
And speaking of statistics, we compare in our study, the intensity distribution of individual pixels for each image class. And through this comparison, we were able to draw some interesting conclusions about the classes and some common characteristics between the classes. For more information, we invite you to look at our GitHub repository where you are invited as a reviewer.

### MODELS IMPLEMENTED FOR THIS PROJECT

### 1.	Base models: KNN, Logistic Regression, Random Forest Classifier
For the project, we decide to set up a progressive methodology to study and better understand the dataset. So, firstly, we build the K-Nearest-Neighbors, Logistic Regression and Random Forest algorithm. And to do this, we used the dataframe containing the intensity of individual pixels for each image class (hmnist_64_64_L.csv). 
The models set up obviously make no sense because the characteristics of an image do not depend on individual pixels but rather on patterns. But we used them as a reference base for the other methods we will set up.
Also, following the use of these models, we studied the intensity density of the tissues of the classes taken separately and then the mode density of each individual image. This allowed us to draw strong conclusions about the classes, and thus to think about the model to implement. For example, we thought that if we use neural networks to classify cancers, these different images within the same class will lead the model to draw wrong decision boundaries.

### 2.	Neural Network
A neural network is a computer model with a layered structure like the network structure of neurons in the brain, with layers of connected nodes. We decided to model it on the dataset we have available to classify tissue types. Let's describe a bit what this convolutional neural network does on our dataset.
We have indeed a data set with 5,000 samples. This dataset will be divided in training, validation, and test data, each one having 64 * 64 pixels of resolution.  Then, using Keras (which is a TensorFlow backend), we will create a convolution neural network. Here is a quick description of what the CNN algorithm does:
- To build a simple CNN, there are 3 types of layers: Convolution, Pooling and Fully-connected. The convolution layer applies a filter (kernel) to transform the input image into another type of image.
-	Pooling, on the other hand, reduces the dimensionality of the image. For example, if our image contains a dark spot of size 5x5, the reduced spot (based on the pooling layer settings) will be reduced to a dark spot of 3x3. 
- The fully connected layer is the typical layer of a multilayer perceiver. Before this layer is added, the N x N matrix must be flattened into an array.
Thereafter, learning in these layers is done by back propagation.

### DATA PRE-PROCESSING AND EXPERIMENTAL METHODOLOGY

After importing the data and libraries needed to read the data and implement the machine learning models, we performed a pre-processing of the images before training and evaluating these models. 

### 1.	Base models 
Before training the data to these models, we needed the individual pixel intensities for each image class using the 64 x 64 pixels resolution. Then we divided the dataset with a proportion of 20% while stratifying it using the pixel value of the class labels.
Thereafter, the models are directly evaluated using a function and the parameters of each model. The function returns the score obtained after the training according to the model/
-	KNN: we use KNeighborsClassifier () by just specifying the number of parallel tasks to execute for the neighbor search (n_jobs= 1)
-	Logistic Regression: we use LogisticRegression () by just specifying the number of jobs to run in parallel (n_jobs= 1) and the solver (solver='lbfgs’)
-	Random Forest Classifier: we use RandomForestClassifier () by specifying the number of jobs to run in parallel (n_jobs=1)
The result is thus given in the form of a confusion matrix which measures the quality of the classification system. Each row corresponds to a real class and each column corresponds to the estimated class.

### 2.	Neural Network
To build the neural network, we make a rescale of the images and a reshape to obtain data whose input size is 64 x 64 pixels (the original images are 150 x 150 pixels). 
Then, we build the sequential model of the following form:
-	Conv2D
The important thing about convolution layers is the pattern they learn. The first convolution layer learns small patterns, the second convolution layer learns larger patterns made up of features from the first layers, etc. When we use this layer, we will be interested in certain aspects of these patterns: the filters (128 then 64), a pattern of size also called kernel, the padding, the activation function, and the input shape of the data.
-	MaxPooling
With the first Conv2D layer we get 128 feature-map, 128 image filters each containing features. After using a Conv2D layer, we have to reduce the result obtained. This is easily done with the MaxPooling2D layer. The MaxPooling is also a convolution layer. It extracts patterns, trends from a data. But where Conv2D extracts features from an image to create feature maps, MaxPooling2D extracts the most important value of each pattern from the feature maps. In fact, it takes as input feature-maps to extract the max value. It keeps only the important information. It allows to gain in precision by keeping only the relevant data and in speed. We did not specify the number of filters because MaxPooling adapts to the previous layer.
-	Flatten
This layer is used to flatten the tensor and reduce its dimension. It takes as input a 3D-tensor and returns a 1D-tensor. The Flatten layer allows to establish a connection between the convolution layers and the base layers
-	Activation functions
Activation functions can be thought of as a set of rules that determine whether a neuron activates/"fires" or not, based on an input or a set of inputs. 
A neuron with a ReLU activation function takes all real values as input(s), but only activates when these inputs are greater than 0. As for SoftMax, it is very interesting because it not only maps our output to a range of [0,1] but also maps each output so that the sum is 1. This is a probability distribution.

The compilation is performed using one single method call called compile model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"]. The compile method requires several parameters. The loss parameter is specified to have type 'categorical_crossentropy'. The metrics parameter is set to 'accuracy' and finally we use the adam optimizer for training the network.
Then we had to indicate the batch_size, i.e., the number of samples in each mini-batch. The maximum is the number of samples for which the gradient descent is accurate, and the learning rate is low, making the code slower. The minimum is 1. 
Next is the number of steps_per_epoch. This one essentially goes through the other samples not considered in the original epoch batch. Typically, we can have steps_per_epoch = train_length/batch_size. Since we've augmented our data with flips, shifts, and rotations, we can multiply them by a few.
The number of epochs is determined by when we see the validation accuracy approaching the training accuracy. When these start to diverge, we will have a model that overfits the data.

#### MODELS RESULTS

### 1.	Base models 
With the basic model, we did not obtain satisfactory results because they are weak and as we said so well at the beginning these models make no sense because the characteristics of an image do not depend on individual pixels but rather on patterns. But we use them as a reference base. 
-	KNN: The score obtains is 0.37
 
-	Logistic Regression: The score obtains is 0.23
 
-	Random Forest Classifier: The score obtains is 0.7
 
Obviously, it’s the unique model of the base model that seems to be interesting to improve for this dataset study. 

### 2.	Neural Network
The score obtains with the neural network it’s more interesting than the other methods. 
It is more complex but also more interesting to better understand our dataset and learn it better. It is not quite stable now (as we can see in the training plot). We still must play with some parameters to optimize it. This includes parameters like batch_size, steps_per_epoch and epoch number. The compilation of the model takes more time, it would be interesting to conduct research on this model and exploit other architectures of the convolution network like the ResNet. 

### MODELS INCONVENIENTS AND PERSPECTIVES

One drawback of the approaches used is that they are based on a defined area of 150 x 150 pixels. It is possible that a certain area contains several different cell/tissue types. For example, it is well known that tumor-immunity/tumor-stroma interactions play an important role in tumor progression. This is because the mucosa itself contains immune cells and multiple other cell types. 
Therefore, one solution would be to use a broader classification criterion instead of trying to refine the tissue categories into 8 different classes. In addition, attention should be paid to image selection for the training and validation datasets: images with multiple cell types should be eliminated to avoid potential misclassification.

### CONCLUSION

At the end of our study, we can say that the implemented models allowed us to understand the structure of the tissues and to discover their individual as well as common features. The convolutional network model set up allowed us to obtain quite perfect even if it is not quite stable. It could obviously be improved by playing with some parameters.  It could also be interesting to look at other architectures to build the convolutional network. For example, the ResNet architecture and automated machine learning models (AutoML). 
However, the implemented CNN model is effective in classifying classes that have a distinct pixel distribution pattern. It is very important and interesting to look at the images of individual cell statistics to better learn and draw conclusions on this topic. 
