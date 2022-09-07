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
