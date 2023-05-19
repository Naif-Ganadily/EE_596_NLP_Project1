# EE_596_NLP_Project1
# Sentiment Analysis with Logistic Regression and Deep Learning: A First Deep Dive into NLP
### Prof. Chandra Bhagavatula
### Student: Naif A. Ganadily
This repository contains the implementation of a sentiment analysis model using logistic regression. The project is divided into three parts:


Part 1: Basic implementation of logistic regression using Scikit-Learn.
Part 2: Deep Learning approach to sentiment analysis using PyTorch.
Part 3: Implementation of Logistic Regression with pre-trained Word Embeddings (GloVe).



## Dependencies
- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- NLTK
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset
The dataset used in this project is a set of movie reviews labeled as either positive or negative. It is divided into three parts: training, validation, and testing sets.

## Preprocessing
The text data is preprocessed by:

1. Converting the text to lowercase.
2. Removing punctuation.
3. Tokenizing the text.
4. Removing stop words and empty strings.
5. Lemmatizing tokens.

## Models
## * Part 1: Basic Logistic Regression
In this part, a basic logistic regression model is implemented using Scikit-Learn. The model takes a Bag of Words (BoW) representation of the movie reviews and predicts the sentiment of the movie review (positive or negative).

## * Part 2: Deep Learning Approach with PyTorch
In this part, I have used PyTorch to create a deep learning-based logistic regression model for sentiment analysis. The model is trained using backpropagation over a set number of epochs. During training, cross-validation is performed to monitor the validation loss and the best model is saved. The performance of the model is then evaluated on the validation and test datasets.

## * Part 3: Logistic Regression with Pretrained Word Embeddings
In the third part, pretrained word embeddings (GloVe) are used to represent the movie reviews. Each review is represented as the sum of its word vectors. A logistic regression model is applied on this representation, and its performance is evaluated on the validation and test datasets.

## Evaluation Metrics
The models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall

## Results
Training, validation loss and accuracy, precision, and recall are tracked for each model. The models are tested on the test dataset, and the confusion matrix, precision-recall curve, and ROC curve are visualized.

## Conclusion
The project provides a comprehensive look into implementing sentiment analysis using traditional machine learning and deep learning approaches, illustrating the entire pipeline from preprocessing the text data, converting it into numerical format, training the model, to evaluating its performance. Both traditional logistic regression and deep learning approaches deliver competitive results, and the usage of pretrained word embeddings in the third part further enhances the model's performance.
