The quality evaluation of agricultural products plays a crucial role in ensuring
consumer satisfaction and market competitiveness. The objective of this project is to
develop a robust and efficient machine vision system coupled with machine learning
techniques for classifying black pepper samples based on their quality attributes. The
process begins by collecting four types of black pepper samples from the market and
capturing 50 images of each sample. These images are processed to extract pixel values
along with additional features such as RGB and grayscale values. A dataset is
constructed using these features, and principal component analysis (PCA) is applied for
dimensionality reduction and visualization. The dataset also includes an additional
column representing the piperine content (PIP) of each sample. Several regression
models, including linear regression, ridge regression, and ensemble methods like
random forest and gradient boosting, are employed to predict the PIP values. The
performance of each model is evaluated using metrics such as mean squared error and
R-squared to identify the best-performing model. In addition to traditional machine
learning models, this study explores the use of convolutional neural networks (CNNs)
for image classification. A CNN architecture is trained on the image dataset to classify
black pepper samples into different quality categories. Transfer learning is leveraged by
using a pre-trained  model for feature extraction, followed by additional
fully connected layers for classification. The effectiveness of the proposed approach is
demonstrated through comprehensive experiments and evaluations. The developed
system offers a reliable and automated solution for the quality evaluation of black
pepper, providing valuable insights for stakeholders in the agricultural industry
