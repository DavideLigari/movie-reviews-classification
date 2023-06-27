# movie-reviews

This repository contains a comprehensive analysis of the steps taken to develop a machine learning model for classifying film reviews. The analysis focuses on the Andrew Maas's Large Movie Review Dataset, which consists of 50,000 labeled film reviews as either positive or negative.

## Overview

The goal of this project is to develop a machine learning model that can accurately classify film reviews as positive or negative. The work is divided into several stages, as outlined below:

1. Preprocessing Phase:
   - Creation of a vocabulary: All words in the dataset are extracted to form a vocabulary.
   - Bag-of-Words (BOW) approach: The number of occurrences of each word in each review is counted, creating a BOW representation.

2. Model Training:
   - Different models are created and compared using various preprocessing techniques.
   - Preprocessing techniques include stemming and the removal of common and meaningless words from the vocabulary.
   - The BOW representation is used to train the models.

3. Model Evaluation:
   - The trained models are evaluated using appropriate evaluation metrics.
   - Different classifiers, such as logistic regression and multinomial Naive Bayes, are used and their performance is compared.

## Dataset

The dataset used in this study is Andrew Maas's Large Movie Review Dataset. It consists of 50,000 film reviews labeled as positive or negative. The dataset provides a reliable foundation for training and evaluating the machine learning models.

## Usage

If you wish to use this code or replicate the analysis, please follow these steps:

1. Download the Andrew Maas's Large Movie Review Dataset from [source URL].
2. Ensure that the necessary dependencies and libraries are installed (provide a list if applicable).
3. Run the preprocessing script to create the vocabulary and generate the BOW representation.
4. Train the machine learning models using the generated BOW representation and different preprocessing techniques.
5. Evaluate the models using appropriate evaluation metrics.
6. Compare the performance of different classifiers.
7. Analyze the results and draw conclusions based on the findings.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact me via email at davide.ligari01@gmail.com

Thank you for your interest in this project!
