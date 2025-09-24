# End-to-End Machine Learning Project: Credit Card Fraud Detection
This repository contains a complete, end-to-end machine learning project focused on detecting fraudulent credit card transactions using a Random Forest Classifier. The project showcases a full data science workflow, from data exploration and preprocessing to model training, evaluation, and hyperparameter tuning, with a strong emphasis on handling real-world challenges like severe class imbalance and preventing data leakage.

1. Project Overview
The goal of this project is to build a robust machine learning model that can accurately identify fraudulent credit card transactions. Given the nature of financial data, this is a classic binary classification problem with a significant class imbalance, as fraudulent transactions are extremely rare compared to legitimate ones.

This notebook demonstrates a practical and effective approach to solving this problem using a RandomForestClassifier, highlighting best practices that are critical in a real-world financial or quantitative analysis setting.

2. The Machine Learning Workflow
The project is structured to follow a standard, rigorous data science methodology:

2.1. Exploratory Data Analysis (EDA)
The first step was to understand the dataset's characteristics. Key findings from the EDA include:

The dataset contains 29 anonymized numerical features (V1, V2, etc.) and a target Class variable.

There were no missing values, which simplified the cleaning process.

A critical finding was the severe class imbalance: fraudulent transactions represent only about 1% of the entire dataset. This immediately established that accuracy is a misleading metric and that a more nuanced evaluation approach would be required.

2.2. Data Preprocessing & Preventing Data Leakage
To prepare the data for the model, a crucial preprocessing pipeline was established with a focus on preventing data leakage.

Train-Test Split: The data was split into training and testing sets before any scaling was performed. The stratify=y parameter was used to ensure that the small minority class was represented in the same proportion in both the train and test sets.

Feature Scaling: StandardScaler was used to standardize the features. Crucially, the scaler was fitted on the training data only and then used to transform both the train and test sets. This is a critical best practice to prevent information from the test set from "leaking" into the training process.

2.3. Model Training: Random Forest
A Random Forest Classifier was chosen due to its robustness, high performance, and ability to handle complex, non-linear relationships in data.

To combat the severe class imbalance, the model was instantiated with the class_weight='balanced' parameter. This setting automatically adjusts the weights inversely proportional to class frequencies, forcing the model to pay significantly more attention to the minority (fraud) class during training.

2.4. Model Evaluation
Given the class imbalance, the model's performance was evaluated using metrics that provide a clearer picture than simple accuracy:

Confusion Matrix: To see the raw counts of true positives, true negatives, false positives, and (most importantly) false negatives.

Classification Report: To analyze Precision, Recall, and F1-Score, especially for the minority 'Fraud' class. The primary goal was to maximize Recall for the fraud class, which answers the question: "Of all the actual fraudulent transactions, what percentage did our model correctly identify?"

2.5. Model Interpretation: Feature Importance
A key advantage of Random Forests is their built-in ability to calculate feature importance. A bar chart was generated to visualize the top 15 features that the model found most predictive of fraud. In a real-world financial institution, this information is invaluable for explaining the model's decisions to stakeholders and fraud analysts.

2.6. Hyperparameter Tuning
To optimize the model for the best possible performance, RandomizedSearchCV was used. This process systematically searches for the optimal combination of hyperparameters (like the number of trees, their max depth, etc.). The search was configured to optimize for the recall score, directly aligning the tuning process with our primary business goal of catching as many fraudulent transactions as possible.

3. Key Results & Findings
The final, tuned Random Forest model performed exceptionally well, demonstrating its ability to effectively identify fraudulent transactions despite the severe class imbalance. The model achieved a high Recall score for the fraud class, successfully flagging a majority of the fraudulent transactions in the unseen test set while maintaining reasonable precision.

This project demonstrates that by using the right techniques—such as stratified splitting, careful preprocessing, class weighting, and proper evaluation metrics—it is possible to build a highly effective and reliable fraud detection model.

4. How to Run This Project
Clone this repository to your local machine.

Ensure you have the necessary Python libraries installed. You can install them using the provided requirements.txt file:

pip install -r requirements.txt

Open and run the Jupyter Notebook fraud_detection_masterclass.ipynb in a Jupyter environment.

5. Technologies Used
Python

Pandas for data manipulation and analysis.

NumPy for numerical operations.

Scikit-learn for data preprocessing, modeling, and evaluation.

Matplotlib & Seaborn for data visualization.

Jupyter Notebook for interactive development.
