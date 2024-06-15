This repository contains a GUI application for performing sentiment analysis using three different models: Logistic Regression, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM). The application allows users to load a dataset, train models, view performance metrics, and predict sentiment on new text inputs.

Features:
 Load and preprocess a dataset for sentiment analysis.
 Train and evaluate Logistic Regression, SVM, and LSTM models.
 Display performance metrics including accuracy and classification reports.
 Visualize confusion matrices for each model.
 Compare model accuracies.
 Predict sentiment for custom text inputs.

Usage:
 Running the Application
 To run the application, execute the following command:

 bash:
 python app.py
 Application Interface
 Load Dataset: Click the "Load Dataset" button to select a CSV file containing the dataset. The dataset should have text and sentiment columns.
 Train Models: Use the buttons in the "Train Models" section to train Logistic Regression, SVM, or LSTM models.
 View Performance: After training, view performance metrics and confusion matrices for each model.
 Model Comparison: Click the "Show Model Comparison" button to view a bar chart comparing model accuracies.
 Predict Sentiment: Enter custom text in the provided input box and click "Predict" to get sentiment predictions from all three models.
 Dataset Requirements
 
The dataset should be in CSV format with the following columns:

text: The text data to be analyzed.
sentiment: The corresponding sentiment labels (e.g., positive, neutral, negative).

text,sentiment
"I love this product!",positive
"This is the worst service ever.",negative
"It's okay, not great but not bad either.",neutral
Application Preview


Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
