# ANDI Alziheimers Classifier using ConvNeXt

## 1. Introduction

ConvNeXt is a new and updated CNN architecture that combines elements of ResNet with design features from Vision Transformers (ViTs). While more tradititonal CNN's like ResNet are efficient and scalable, ViT's are strong at dealing with larger scale datasets, and more specifc normalization/regularisation techniques. ConvNeXt provides the best of both of these architectures combining features from both approaches.

## 2. Model 

### 2.1 Model innovations and strengths

#### Patchify Stem:
    - Replaces standard 7*7 convolution from other CNN's to a 4*4 convolution of stride 4. This is adapted from patch embedding of ViTs
    - Creates non-overlapping patches, increasing effeciency and performance over other CNNs

#### Stagewise Architecture:
    - Four stage structure, processes progressively lower-res feature maps

#### Inverted Bottleneck Blocks:
    - Expansion occurs before convolution
    - Uses GELU instead of RELU to match transformer architectures and for better performance

#### Layer Normalization:
    - Uses layer instead of batch normalization for tranformer compatibility

#### Depthwise Seperable Convolutions
    - Makes the network deeper and wider without increasing computational cost

### 2.2 Model Architecture

The following diagram demonstrates the structure of the ConvNeXt architecture. Each block can be charactrised by a depthwise convolution, layer normlaization, pointwise convoltutions and GELU activation

![alt text](ConvNeXt-structure.webp)

#### Stem
    - Takes a standard 224*224*3 image as input and convolves using a 4*4 kernel and stride 4
    - The output is a 56×56×96 feature map, where each pixel represents a learned embedding of a 4×4 region of the image.

#### Stage 1
    - Applies Depthwise Convolution on the 56×56×96 feature map from the stem
    - Applies layer Normilisation to stabilize training
    - Extracts low-level textures

#### Stage 2-3
    - Downsamples further to 28*28*192 and then to 14*14*384, getting more mid level feature deatils

#### Stage 4
    - Fetaure extraction at 7*7*768, 
    - Feature map is averaged, aggregating information across the entire image 
    - Vector is normalized and passed through a softmax function which creates class probabilities

#### Stage 5
    - Classification of image

## 3. Dataset

This implementation of ConvNeXT uses the ANDI Alizheimers data set of brain MRI data. This data has been processed into greyscale and has been sorted into train and test data based on AD (non-healthy) and NC (healthy) data. Each image has been rezised into the default format for ConvNeXt (224*224) and has then been converted to a torch tensor. In order to improve the qulaity of model training, images in the dataset have been randomly flipped, rotated, and translated to esnure that the model is robust and will not overfit the training dataset. The train data was further partitioned into a smaller validate set for hyperparameter tuning.
![A sample image from the dataset](218391_78.jpeg)

## 4. Training
The model is set to train using binary class classification by sorting images into healthy and unhealthy data. Each iteration of training is a forward pass of the model, and has been refined using parameters of training.

### 4.1 Parameters of training

#### Learning rate
    - Reflects the rate at which model ajusts hyperparameters in response to error or varied training data.

#### Weight decay
    - Reduces tendency towards overfitting by penalizing large weights. Is regulated using AdamW in order to optimize for varied losses

#### Batch size
    - Number of samples processed together during a single iteration

### 4.2 Indicators of accuracy

#### Accuracy
    - Indicates the proportion of correctly sorted images compared to the train/validate classifications

#### Cross-entropy loss
    - Calculated as BCE=−N1​Σi=1N​(yi​.log(pi​)+(1−yi​)log(1−pi​)), where N is no. samples, yi is true label and pi is predcited probability of class 1 for sample i
    - Measures the closeness of a models predictions to that of the correct classification

### 4.3 Training Results
The following graphs plot each epoch against the accuracy and cross entropy loss respectively of each of the train and validation data sets. As is shown, the validation dataset tends to

### 4.4 Challenges and how they were overcome

## Predict
Trains loads and evaluates the ConnvNeXt model with the given training data, and then evaluates the test set for training and validation loss.












 









