# Spam-SMS-Detection

📩 Spam SMS Detection

🚀 Project Overview
This project focuses on detecting and classifying SMS messages as either Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Machine Learning techniques.
It helps in filtering out unwanted and malicious messages, providing a safer communication environment.

🛠 Tech Stack
Programming Language: Python 🐍

Libraries:

Pandas, NumPy

Scikit-learn

NLTK

Matplotlib, Seaborn

WordCloud

Joblib

📚 Features
Text Preprocessing:

Lowercasing

Removing punctuation, numbers

Removing stopwords

Stemming words

Feature Extraction:

TF-IDF Vectorization (Top 5000 features)

Model Training:

Multinomial Naive Bayes

Logistic Regression

Random Forest Classifier

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Visualization:

Word Clouds for Spam and Ham messages

Model Saving:

Best performing model saved as .pkl file for future predictions

📈 Model Performance
Models were evaluated on a test set (20% split).

Achieved high accuracy and F1-score across all models.

Best performing model: <Model Name> (update this based on your result)

🔥 How it Works
Preprocess the SMS message.

Transform the message using the trained TF-IDF vectorizer.

Predict using the best saved model.

Classify as Spam (1) or Ham (0).

📝 Installation Guide
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/spam-sms-detection.git

# Navigate to the project directory
cd spam-sms-detection

# Install dependencies
pip install -r requirements.txt

# Download necessary NLTK data
python -c "import nltk; nltk.download('stopwords')"
⚙️ Usage
python
Copy
Edit
from joblib import load
import preprocessing_function  # make sure your preprocessing function is available

# Load saved model and vectorizer
model = load('spam_detection_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

# Sample message
message = "Congratulations! You've won a $1000 gift card. Call now to claim!"

# Preprocess
processed_message = preprocess_text(message)

# Transform
X_new = vectorizer.transform([processed_message])

# Predict
prediction = model.predict(X_new)
print("Spam" if prediction[0] == 1 else "Ham")
📊 Visualizations
Confusion Matrix plots

Word Cloud of Spam messages

Word Cloud of Ham messages

Example:


Spam Word Cloud	Ham Word Cloud
