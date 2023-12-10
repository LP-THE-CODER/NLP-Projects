# Sentiment Analysis with Naive Bayes Classifier

## Overview

This repository contains a machine learning model for sentiment analysis using a Naive Bayes classifier. The model is trained on a dataset of restaurant reviews and can predict whether a given review is positive or negative.

## Dataset

The dataset used for training and testing is "Restaurant_Reviews.tsv". It includes reviews along with their corresponding labels (1 for positive, 0 for negative).

## Data Preprocessing

The reviews undergo a series of preprocessing steps, including:
- Removal of special characters
- Conversion to lowercase
- Tokenization
- Removal of stopwords
- Stemming

The preprocessed reviews are then transformed into a Bag of Words model using the CountVectorizer from scikit-learn.

## Model Training

The Naive Bayes classifier (MultinomialNB) is used for training the model. The dataset is split into training and testing sets, and the model is evaluated based on accuracy, precision, recall, and a confusion matrix.

## Hyperparameter Tuning

The alpha hyperparameter of the Naive Bayes classifier is tuned for better performance.

## Predictions

A function `predict_sentiment` is provided to predict the sentiment of new reviews. Examples are included in the code for demonstration.

## How to Use

To use the sentiment analysis model, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/LP-THE-CODER/sentiment-analysis.git
