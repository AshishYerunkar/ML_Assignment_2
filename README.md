# Spam Classification Using Machine Learning

## 1. Problem Statement

Email spam is an unwanted or unsolicited message that may contain advertising, misleading information, phishing attempts, or other undesirable content. Automatically distinguishing spam from legitimate email is therefore an important binary classification problem.

The objective of this project is to develop and compare multiple machine-learning classification algorithms for identifying whether an email is spam or non-spam. Five classification models are trained and evaluated on the same dataset using a common train-test split.

The implemented models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

Model performance is evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also provided to upload labelled test data, select a classification model, view evaluation metrics, inspect the confusion matrix and classification report, and compare the performance of all implemented models.


## 2. Dataset Description

The project uses the **Spambase** dataset from the UCI Machine Learning Repository.

The dataset represents email messages using numerical characteristics extracted from their contents.

### Dataset characteristics

- Number of instances: **4,601**
- Number of predictor features: **57**
- Target variable: **spam**
- Classification type: **Binary classification**
- Missing values: **0**

### Target classes

- `0` = Non-spam
- `1` = Spam

The complete dataset contains:

- **2,788 non-spam observations**
- **1,813 spam observations**

The predictor variables include word-frequency features, character-frequency features, and capital-letter sequence characteristics.

Examples include:

- `word_freq_make`
- `word_freq_address`
- `word_freq_all`
- `char_freq_$`
- `char_freq_#`
- `capital_run_length_average`
- `capital_run_length_longest`
- `capital_run_length_total`

A stratified 80/20 train-test split was used so that the class proportions were preserved.

### Train-test split

Training set:

- 3,680 observations
- Non-spam: 2,230
- Spam: 1,450

Test set:

- 921 observations
- Non-spam: 558
- Spam: 363

The supplied `test_data.csv` contains the 921 held-out observations used for final evaluation in the Streamlit application.


## 3. GitHub Repository Link

**GitHub Repository:** To be added after repository creation.


## 4. Live Streamlit Application

**Streamlit Application:** To be added after deployment.


## 5. Models Used and Performance Comparison

All models were evaluated on the same held-out test dataset using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9240 | 0.9714 | 0.9347 | 0.8678 | 0.9000 | 0.8403 |
| Decision Tree | 0.9251 | 0.9199 | 0.9129 | 0.8953 | 0.9040 | 0.8427 |
| K-Nearest Neighbors | 0.9055 | 0.9604 | 0.9059 | 0.8485 | 0.8762 | 0.8011 |
| Naive Bayes | 0.8111 | 0.9449 | 0.6879 | 0.9532 | 0.7991 | 0.6594 |
| Random Forest | **0.9620** | **0.9899** | **0.9581** | 0.9449 | **0.9515** | **0.9203** |


## 6. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9240 and a high AUC of 0.9714. Its precision of 0.9347 indicates that most emails predicted as spam were actually spam. Its recall of 0.8678 was lower than its precision, indicating that some spam messages were classified as non-spam. Overall, the model provided strong and balanced classification performance.

### Decision Tree

The Decision Tree achieved an accuracy of 0.9251, which was slightly higher than Logistic Regression. Its recall of 0.8953 and F1 score of 0.9040 were also slightly higher than Logistic Regression. However, its AUC of 0.9199 was the lowest among the five models, indicating weaker overall ranking ability compared with the other classifiers.

### K-Nearest Neighbors

K-Nearest Neighbors achieved an accuracy of 0.9055 and an AUC of 0.9604. The model showed good precision of 0.9059 but had lower recall of 0.8485. Its F1 score of 0.8762 and MCC of 0.8011 indicate reasonable classification performance, although it did not outperform Logistic Regression, Decision Tree, or Random Forest on the overall evaluation.

### Naive Bayes

Gaussian Naive Bayes achieved the highest recall among all models at 0.9532. This indicates that it successfully identified a very high proportion of actual spam messages. However, its precision was only 0.6879 and its accuracy was 0.8111. Therefore, although it missed relatively few spam messages, it classified more legitimate emails as spam than the other models. Its MCC of 0.6594 was also the lowest among the models.

### Random Forest

Random Forest achieved the best overall performance. It obtained the highest Accuracy (0.9620), AUC (0.9899), Precision (0.9581), F1 Score (0.9515), and MCC (0.9203). Its recall of 0.9449 was also very high and was second only to Naive Bayes. These results indicate that Random Forest provided the strongest balance between detecting spam and avoiding false spam classifications.


## 7. Overall Winner

**Random Forest is the overall best-performing model for this dataset.**

It produced the highest values for five of the six evaluation metrics:

- Accuracy: **0.9620**
- AUC: **0.9899**
- Precision: **0.9581**
- F1 Score: **0.9515**
- MCC: **0.9203**

Naive Bayes achieved the highest recall of **0.9532**, compared with Random Forest's recall of **0.9449**. However, Random Forest demonstrated substantially stronger overall performance and better balance across all evaluation measures.


## 8. Streamlit Application Features

The Streamlit application provides:

- Upload of labelled test data in CSV format
- Validation of uploaded predictor and target columns
- Classification model selection using a dropdown
- Evaluation of the selected model
- Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Confusion matrix
- Classification report
- Performance comparison of all implemented models
- Identification of the model with the highest MCC


## 9. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── train_models.py
    ├── train_decision_tree.py
    ├── train_knn.py
    ├── train_naive_bayes.py
    ├── train_random_forest.py
    ├── evaluate_all_models.py
    └── create_test_data.py

## 10. Running the Application Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload `test_data.csv` through the application and select a model from the dropdown to view its evaluation results.
