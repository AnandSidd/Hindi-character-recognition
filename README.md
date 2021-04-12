# Hindi Handwritten character recognition

The original dataset has been taken from the below mentioned site.
The link for the data - https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset
Since the dataset was huge, I haven't uploaded it here.

This aims at classifying handwritten devanagri letters into 36 classes.
The dataset consisted of 61200 32X32 images. Each image was greyscaled. The entire data was divided into three parts training, validation and test set.
The model has been currently trained only for the consonants, vowels can also be added.

# Preprocessing the Image

The input image of the hindi handwritten word is taken as input and preprocessed.
The preprocessing involves:
 =>Image noise removal - We have preprocessed our input image in such a way that noises due to variable lighting conditions, shadows, various noisy blobs are removed.
 =>Image deskewing- If the word in our input image is skewed or written at an angle, we deskew the word to bring it to proper alignment. ie. if our word is at an angle of 45 degree to the x axis we change the angle to 0 degree.//
 => Removing SHIROREKHA- The line on which our hindi word is written is called shirorekha. In our Image preprocessing, We have prepared an algorithm to remove shirorekha, after removing shirorekha, we parse through individual words to detect them and output the string in the result.//
Each character has been segmented and then predicted by our model.

The file dataprep.py prepares the dataset by dividing them into train and test and converting into csv files.
image_processing.py contains all the image preprocessing part.
training.py contain the the model training part which uses convolutional neural network for developing the model.
main.py takes image as input and prints the predicted word as output.

All the required python packages are given in the requirement.txt file.
