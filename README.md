# CNN-Image-Classification-and-Denoising
This GitHub repository showcases a project that involves building a convolutional neural network (CNN) classifier for the Fashion-MNIST dataset using TensorFlow and Keras. The main objective of this project is to accurately classify fashion images into various categories.

The project includes the following key components:

1. Dataset Preprocessing: The Fashion-MNIST dataset is preprocessed by scaling pixel values, adding random noise, and performing one-hot encoding on the labels. These steps ensure that the data is in the appropriate format for training and testing the models.

2. CNN Classifier: A CNN classifier is developed using TensorFlow and Keras. This classifier is trained on the preprocessed Fashion-MNIST dataset to learn and identify patterns in the images. Early stopping and validation accuracy monitoring techniques are employed to prevent overfitting. The trained classifier achieves an impressive classification accuracy of 90% on clean test images.

3. Denoising Autoencoder: In addition to the CNN classifier, a CNN autoencoder is implemented to denoise the images. The autoencoder is trained to remove noise from the input images, enhancing their quality and reducing the impact of noise on the classification task.

4. Fine-tuning with Denoised Images: The denoised images are then used to fine-tune the CNN classifier. By utilizing the denoised images as input, the classifier achieves a classification accuracy of 85%, demonstrating the effectiveness of the denoising autoencoder in improving classification performance.

The repository provides the complete source code, including the implementation of the CNN classifier, denoising autoencoder, and necessary preprocessing steps. It also includes instructions on how to train and evaluate the models. Developers and machine learning enthusiasts can leverage this repository to study, and  reproduce the project's findings
