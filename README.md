# Brain-tumor-analysis
MRI technology is used to identify brain cancers because MRI scans have great resolution and may clearly depict the structure, size, and location of brain tumours. BraTS 2015 dataset is used for brain tumor filtering, segmentation and classification.
# Filtering
Most of the MRI images consists of more noise to remove that noise we are using Median filtering and mean filtering algorithms. Both the filtering algorithms used to remove noise from an image but the quality of the output image varies. Median filtering has high accuracy than mean filtering algorithm. So the quality of median filtered image is very high compared with mean filtered image.

Procedure :
           
           Importing files and uploading the dataset
           
           Unzipping the dataset
           
           Importing libraries
           
           Training the data by assigning the batch size of 9
           
           Plotting the trained dataset images if Tumor exist=1 orelse 0
           
           Applying median filtering to the random image
           
           Applying mean filtering to the random image
           
           Calculating the PSNR and MSE values

![image](https://user-images.githubusercontent.com/107994772/188938970-5bf7611c-ef7f-4206-86e4-efa23c693135.png)
# Segmentation
Segmentation is the process used to separate the image into different segments in the form of pixels. for segmentation purpose we are using VGG16 and ResNet50 algorithms. VGG16 is a object detection algorithm and it is used here for semantic segmentation purpose. It is used to enhance the performance of unet. It consists of 16 layers typically 13 convolutional layers, five Max Pooling layers, and three Dense layers but vgg16 contains only 16 weighted layers. ResNet50 is a convolutional neural network that contains 50 layers. In resnet50 we can train millions of images. 

Among both the segmentation algorithms VGG16 has high accuracy and it is used for classification of images too.

Procedure :
           
           After filtering the filtered output can be segmented to identify whether the tumor exist or not based on the characteristics
           
           Applying vgg16 algorithm and plotting the segmented image
           
           Applying resnet50 algorithm and plotting the segmented image
           
           Calculating the accuracy for both the segmented images and analyzing the best algorithm

final segmented image of Vgg16 algorithm

![image](https://user-images.githubusercontent.com/107994772/188947604-79f25ec0-ab74-49f7-a481-5644574d9dd0.png)

# Classification

         
