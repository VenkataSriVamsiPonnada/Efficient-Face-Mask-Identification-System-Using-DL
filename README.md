When people are moving in crowded places, there a lot of chances for increase of spread of corona virus. If people wear mask and follow all certain preventive measures strictly, then the spread can be decreased and avoided. But, people are not following and become a reason for the spread unintentionally. To avoid it, people need to be alerted and warned for disobeying the rules. It is difficult to check everyone when moving in the crowd. So, a model needs to be trained to identify the persons without mask and inform authorities to warn them.
3.2	Solution:
For the problem, a model which automatically scans the faces and classifies the unmasked is required. Then, that information need to be sent to authorities. A deep learning model can be trained in such a way as mentioned above for mask detection. That model is called as our Efficient face mask identification system.
3.3	Task Analysis:

Face mask Identification System can perform following tasks:

	Capture the faces:

The system captures the faces from the images captured using the camera.

	Feature Extraction:

From the faces captured, features are extracted using the trained algorithm

	Classification:

Based on those features, the faces are classified as masked or unmasked

	Informing to authorities:

All the unmasked people are identified and that information is send to authorities to warn them.
3.4	Process Model:
The proposed method labels the images with or without mask along with the accuracy. The features are extracted and classified using InceptionV3 and Xception of transfer learning.
 
Each model has its own features and accuracy. The below diagram illustrates the process flow of the proposed system.




Figure 3.4.1: Process flow of the proposed system


3.5	Requirement Specification:
1.	Software Requirements:
	Visual Studio Code
	Google colab
2.	Functional Requirements :
i.	Dataset:
	Must have unbiased dataset
	Must have over 1500+ images in both classes
	Dataset must not reuse the same images in training and testing phases
ii.	Detector:
	Capture the faces
	Extract the features
	Image classification
	Informing to authorities
3.	Non-functional Requirements:
	All the data must be secured
	Time is required to develop the complete project
4.	System Requirements:
	Able to load the mask detection and classifier model
	Able to detect faces in images or video stream
	Able to extract features from the face image
 
	End position of the face must be fit inside the webcam frame and must be clear to camera
	Able to detects masks in .jpg, .png , .gif format images
	Result must be viewed by the showing the probability along with label as Mask
or No Mask
3.6 Use case Modelling:
A UML use case diagram is the primary form of system/software requirements for a new software program underdeveloped. Use cases specify the expected behavior, and not the exact method of making it happen. Use cases once specified can be denoted both textual and visual representation. A key concept of use case modeling is that it helps us design a system from the end user's perspective. It is an effective technique for communicating system behavior in the user's terms by specifying all externally visible system behavior.
Use case diagrams can be used for requirement analysis and high level design model the context of a system,reverse Engineering,forward Engineering.




Figure 3.6.1: Use Case Diagram of InceptionV3 Model
 

 



Figure 3.6.2: Use Case Diagram of XceptionV3 Model

Figure 3.6.3: Use Case Diagram of Web Interfacing
 
4.	METHODOLOGY
4.1	Dataset:
The algorithms are tested on a dataset which consists of various images of faces with mask and without mask. Some of the sample images from the dataset are shown below.
 	 

Figure 4.1.1: Images without mask

Figure 4.1.2: Images with mask

As the data is collected from different sources, the data may not be similar in terms of quality,shape,resolution etc. Because of this, it is hard to extract the features and perform classification. So, image pre-processing and augmentation needs to be done to make all the images look similar in terms of different parameters. Data augmentation includes different rotation,flipping,shearing,etc. At last, all the images are converted into 224*224 size.

Class Label	Sample Count
Faces with mask	1900
Faces without mask	2165
Total	4065
Table 4.1: Classes of images and count of their samples
 
4.2	Algorithms:
There are many deep learning models like VGG16, VGG19, ResNet50, Inception V3 and Xception which can be used for face mask detection. Our proposed system uses InceptionV3 and Xception to build the model. These models are trained on large datasets and extracts features from images and classifies them. In this work, we have compared two algorithms to find the best and optimum model for the problem raised.
InceptionV3:

InceptionV3 is a convolutional neural network which is used for image analysis and object detection. It can load a million size dataset and 48 layers deep. The weights in this model are smaller than other models like VGG and ResNet. It is a multilevel feature extractor.

In this model, each layer performs different operations and forwards the output to the next layer. Each layer performs convolution, avg pooling, max pooling,concatenation. InceptionV3 does 1*1,3*3,5*5 convolution transformation and then a 3*3 max pooling. All that data is filtered and concatenated and passed to the next layer. Every time a filter is added, all the inputs need to convolve to get a single output.


Figure 4.2.1: Architecture of InceptionV3 model
 

Convolution is a process of applying filters to an input which results in an activation. It is a mathematical combination of 2 functions to produce a third function.A convolution converts all the pixels in its receptive field into a single value. For example, if you would apply a convolution to an image, you will be decreasing the image size as well as bringing all the information in the field together into a single pixel. The final output of the convolution layer is a vector. Convolutional layer, as mentioned above this layer consist of sets of Filters or Kernel. They have a key job of carrying out the convolution operation in the first part of the layer.
The filters take a subset of the input data.The operations performed by this layer are linear multiplications with the objective of extract the high-level features such as edges, from the input image as a convolution operational activity.Since convolutional operation at this layer is a linear operation and the output volume is obtained by stacking the activation maps of all filters along the depth dimension. Linear operation mostly involves the multiplication of weights with the input actually same as in traditional neural network. The mathematical equation is input+filter/kernel→ feature map.
Convolution networks are not just limited to only one convolution layer. The first layer is responsible for capturing the low-level features such as colour, edges, gradient orientation etc.

	Conventional Convolution Layer – This layer receives a single input which is a feature map and it computes its output by convolving filters across the feature maps from the previous layer.
	Dynamic Convolution Layer – This layer receives two input, first one as a feature map from the previous layer and second one a filter.

Figure 4.2.2:Example of Convolution layer
 
Average pooling is done after the convolution layer. Pooling is a process of reducing the spatial resolution without changing the 2D representation. Average pooling returns the average of all the values from the portion of images. This method smooths out the image and is used when the focus is on the lighter pixels. Example of avg pooling is shown below.
 

Figure 4.2.3:Example of average pooling Average Pooling Using Keras:
import numpy as np
from keras.models import Sequential
from keras.layers import AveragePooling2D # define input image
image = np.array([[2, 2, 7, 3],
[9, 4, 6, 1],
[8, 5, 2, 4],
[3, 1, 2, 6]])
image = image.reshape(1, 4, 4, 1)
# define model containing just a single average pooling layer model = Sequential(
[AveragePooling2D(pool_size = 2, strides = 2)]) # generate pooled output
output = model.predict(image) # print output image
output = np.squeeze(output) print(output)

Max Pooling is also a type of pooling which is used when the focus is on the brighter pixels. It is completely opposite to avg pooling. This method returns the maximum value from the portion of images covered by kernel.InceptionV3 uses max pooling. Example is shown in the below figure.
 

 
Figure 4.2.4:Example of max pooling Max Pooling Using Keras:
import numpy as np
from keras.models import Sequential from keras.layers import MaxPooling2D # define input image
image = np.array([[2, 2, 7, 3],
[9, 4, 6, 1],
[8, 5, 2, 4],
[3, 1, 2, 6]])
image = image.reshape(1, 4, 4, 1)
# define model containing just a single max pooling layer model = Sequential(
[MaxPooling2D(pool_size = 2, strides = 2)]) # generate pooled output
output = model.predict(image) # print output image
output = np.squeeze(output) print(output)

Xception:
It is a convolution neural network which is defined as extreme inception.It is 71 layers deep.Classify is used in this model to classify the images. It has similar features like InceptionV3 but the inception modules are replaced by depthwise separable convolutions. Xception deals with the depth dimensions rather than spatial dimensions. These convolutions are more efficient than the classical convolutions in terms of computation time. InceptionV3 and Xception share the same number of parameters.
 

 
Figure 4.2.5:Architecture of Xception model

Depthwise separable convolution is a conjunction of pointwise and depthwise convolutions. Xception network comprises 3 flows. The data first passes through entry flow and then through middle flow which is repeated 8 times and then through the exit flow. Each flow performs different operations like convolution, max pooling, separable convolution and an activation function.Based on the number of labels used in the model, the activation function varies. Activation function is a node that is placed in between or at the end of the neural network to decide whether the neuron should be eliminated or not.

4.3	Technologies Used:
	OpenCV:
To detect the faces, images need to be captured from the video. The proposed system need to be attached with the CCTV cameras for capturing. In such case, a real-time computer vision is necessary. OpenCV is the one which is suitable to perform that function. It is a real- time computer vision which is open source and a machine learning software library. It was built to provide a common infrastructure for computer vision applications. It is easy to use and to modify the code. It is a package of many useful and optimized algorithms used to
 
detect, recognize faces and objects, track movements etc. It can find similar images from database, can put images together to produce high resolution image.
	TensorFlow:

It is an open source platform for managing all aspects of machine learning system and mainly focuses on training machine learning models. The APIs of tensorflow are hierarchically arranged with the high level APIs built on low level APIs. To define and train the models tf.keras is used which a high level API and it is also useful to make predictions. TensorFlow is an open source library for fast numerical computing. It was created and is maintained by Google and released under the Apache 2.0 open source license. The API is nominally for the Python programming language, although there is access to the underlying C++ API.

Build and train ML models easily using intuitive high-level APIs like Keras with eager execution, which makes for immediate model iteration and easy debugging. Easily train and deploy models in the cloud, on-prem, in the browser, or on-device no matter what language you use. A simple and flexible architecture to take new ideas from concept to code, to state-of-the-art models, and to publication faster. Tensor flow can be used for

	Easy model building

	Robust ML production anywhere

	Powerful experimentation for research

	Streamlit:
It is an open source app framework for machine learning and data science. Streamlit lets you turn data scripts into shareable web apps in minutes. . We can instantly develop web apps and deploy them easily using Streamlit. Streamlit allows you to write an app the same way you write a python code. Streamlit makes it seamless to work on the interactive loop of coding and viewing results in the web app. Use the this command to install pip install streamlit. After installation, run this command to run the app streamlit run
<yourscript.py> .
 
4.4	Behavioural Aspects of the System:
Activity Diagram:
Activity diagram is basically a flowchart to represent the flow from one activity to another activity. The activity can be described as an operation of the system. It is used by developers to understand the flow of programs on a high level. It also enables them to figure out constraints and conditions that cause particular events. A flow chart converges into being an activity diagram if complex decisions are being made.



Figure 4.4.1: Activity Diagram
 
4.5	Flow Chart:
The below figure depicts the flow chart of the system. All the steps done in for the system are clearly mentioned. Those steps are explained in the next chapter. It is started from importing the required and necessary packages and ended with connecting the web which gives the results. As model is trained using two different algorithms, graphs are also plotted.





Figure 4.5.1: Flowchart of the proposed system
 
4.6	Description of steps:
Step-1:Import the necessary packages
from tensorflow.keras.optimizers import Adam:
It is an optimizer that implements the adam algorithm. It is used for designing deep neural networks and is based on adaptive estimation of first and second order moments.
from tensorflow.keras.utils import to_categorical:
To-categorical is a converter which is used to convert a class vector to a binary matrix.
from sklearn.preprocessing import LabelBinarizer:
Label binarizer is used to convert multiclass labels to binary labels by using inverse transform method.
from sklearn.model_selection import train_test_split:
train_test_split is used to divide the dataset into two subsets. It is a technique to calculate the performance of the algorithm.
from sklearn.metrics import classification_report: Classification_report is used to generate the training report. from imutils import paths:
Imutils is a series of functions to perform on an image like translation,rotation,resizing etc. It is also useful to display the matplotlib images with openCV and python. This package is used to change the paths of images into one list under one path.
import matplotlib.pyplot as plt:
Pyplot is a collection of functions in the matplotlib package. It is used to create a figure,create a plotting area,lines etc.
import numpy as np:
Numpy is a python library which is used when working with arrays concept.Creating a namespace for numpy as np.
import tensorflow_hub as hub:
Tensorflow hub is a storage of trained ML models which can be used and deployed anywhere. Creating a namespace for tensorflow_hub as hub.
import tensorflow as tf:
Tensorflow is used to create dataflow graphs which is used to understand how the data is moving through it. It is used for classification,prediction,understanding,creation etc.Creating a namespace for tensorflow as tf.
 
from tensorflow.keras.preprocessing.image import img_to_array:
It is one of the pre-processing techniques applied on dataset to convert the images into array form.
from tensorflow.keras.preprocessing.image import ImageDataGenerator:
It is also one of the pre-processing techniques applied on dataset to perform transformations and normalization techniques during training the data.
from tensorflow.keras.preprocessing.image import load_img:
This function is used to load the image along with some arguments such as grayscale,target_size,color_mode etc.if grayscale is opted as true then image is loaded as grayscale.if target_size is none, then it defaults to original size.
from tensorflow.keras import layers
from tensorflow.keras.layers import Input import argparse:
Argparse is a parser which is the most common and friendly command line interface for command line options and arguments. It gives errors and help messages when invalid arguments are entered.
from tensorflow.keras.applications import InceptionV3:
This function is used to import the InceptionV3 classes and all the functions in it.
from tensorflow.keras.applications import Xception:
This function is used to import the Xception classes and all the functions in it.

