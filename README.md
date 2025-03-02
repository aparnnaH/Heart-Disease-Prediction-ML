# Heart Disease Prediction using Machine Learning

## Overview
This project predicts heart disease using machine learning models trained on the UCI Heart Disease dataset. The goal is to help identify potential heart disease cases early based on health indicators like age, cholesterol, blood pressure, and heart rate.

## Problem Statement
Heart disease is a leading cause of death worldwide, and early detection is critical. This project aims to develop a machine learning model that predicts whether a person has heart disease based on various health metrics.

## Machine Learning Approach
The dataset was cleaned, visualized, and split into training (80%) and testing (20%) sets. Four models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (Best Model)
- Support Vector Machine (SVM)

### Results Summary:
- **Random Forest** performed best with an 84.2% accuracy, balanced precision, recall, and F1-score.
- **Logistic Regression** and **SVM** had decent accuracy but missed some disease cases.
- **Decision Tree** had a bias towards classifying people as healthy, making it less reliable.

### Results & Demo
Confusion matrices and classification reports were used to compare the models. Random Forest showed the best balance between accuracy and minimizing false negatives, making it the recommended model for heart disease prediction.

**Watch the demo video here**: [[Youtube Link](https://youtu.be/_P3N3pFQbqE)]

## Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook

5. Run the notebook to see data preprocessing, model training, and evaluation.

### Dataset
The project uses the UCI Heart Disease dataset from Kaggle: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download)
### Future Improvements
Fine-tuning Logistic Regression and SVM for better recall.
Exploring deep learning approaches for further improvements.
