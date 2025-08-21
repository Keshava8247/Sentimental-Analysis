
 🎭 Sentiment Analysis on IMDB Movie Reviews

This project demonstrates how to build a Sentiment Analysis model that can classify IMDB movie reviews as Positive or Negative. It combines NLP preprocessing techniques with Machine Learning (Logistic Regression) to achieve a solid baseline for text classification tasks.

📌 Project Overview

Movie reviews are a great source of unstructured data that reflect human opinions. The goal of this project is to:

Clean and preprocess raw text reviews.
Convert the text into meaningful numerical features using TF-IDF Vectorization.
Train a Logistic Regression model for binary sentiment classification.
Evaluate model performance with metrics like accuracy, confusion matrix, and classification report.
Test the trained model on new, unseen reviews.

🧹 Data Preprocessing

To prepare text for modeling, the following steps are applied:

Remove HTML tags and special characters.
Convert text to lowercase.
Remove stopwords (e.g., “the”, “is”, “and”).
Tokenize and normalize the words.
Generate a cleaned_review column.
This reduces noise and ensures the model focuses on meaningful words.

🔎 Feature Engineering

TF-IDF Vectorizer: Converts text into numerical feature vectors.
Limited to 5000 features to balance performance and accuracy.
Output: A sparse matrix representation of reviews.

🤖 Model Training

Algorithm: Logistic Regression (liblinear solver).
Reason: Simple, interpretable, and effective for binary classification tasks.
Split: 75% Training, 25% Testing.

📊 Model Evaluation

The model was evaluated using:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

✔️ Accuracy achieved: ~0.88 – 0.90 (depending on the random split).

🧪 Testing on New Reviews

The trained model can predict unseen reviews:
Positive example: “This movie was fantastic with great acting!” → ✅ Positive
Negative example: “Waste of time. Poor script and boring characters.” → ❌ Negative
Neutral example: “It was okay, nothing special.” → Prediction depends on wording.

📦 Saving & Reusing the Model

The trained model and vectorizer can be saved using Joblib:

joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


This allows future predictions without retraining.

🚀 Future Improvements

Experiment with advanced models (e.g., SVM, Random Forest, or Deep Learning with LSTMs/BERT).
Perform hyperparameter tuning.
Expand dataset with more diverse reviews.
Build a simple Flask/Django API to deploy the model as a web app.
