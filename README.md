# EE_596_NLP_Project1
# Sentiment Analysis with Logistic Regression
### Prof. Chandra Bhagavatula
### Student: Naif A. Ganadily
This repository contains the implementation of a sentiment analysis model using logistic regression. The project is divided into three parts:

* Part 1: Basic implementation of logistic regression.
* Part 2: Implementation of logistic regression with bag of words representation.
* Part 3: Implementation of logistic regression with pretrained word embeddings (GloVe).

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
In this part, a basic logistic regression model is implemented using PyTorch. The model takes a simple input feature (e.g., word count) and predicts the sentiment of the movie review (positive or negative).

## * Part 2: Logistic Regression with Bag of Words
In this part, the bag of words representation is used to represent the movie reviews. Each review is represented as a vector with the same length as the vocabulary, with each element in the vector corresponding to the count of a specific word in the review. The logistic regression model is then applied on this representation.

## * Part 3: Logistic Regression with Pretrained Word Embeddings
In this part, pretrained word embeddings (GloVe) are used to represent the movie reviews. Each review is represented as the sum of its word vectors. A linear layer is applied on top of the sentence representation, and logistic regression is performed.

## Evaluation Metrics
The models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall

## Results
Training and validation loss, accuracy, precision, and recall are plotted for each model. The models are tested on the test dataset, and the confusion matrix, precision-recall curve, and ROC curve are visualized.

