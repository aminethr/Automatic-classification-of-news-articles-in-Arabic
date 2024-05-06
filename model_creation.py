import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('arabic'))

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stop words and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # Join the words back into a single string
    return ' '.join(words)

# Function to load data from .txt files
def load_data_from_txt(folder_path, category):
    data = []
    file_names = os.listdir(folder_path)
    for file_name in tqdm(file_names, desc=f'Loading {category}'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            cleaned_text = preprocess_text(text)
            data.append([cleaned_text, category])
    return data

# Define folder path
folder_path = 'C:/Users/M-Tech/Desktop/Automatic classification of news articles in Arabic/Dataset'

# Load data from all folders
data = []
folders = [ 'politics', 'sport', 'Technology' ,'Art', 'Car' , 'Health' , 'Tourism' , 'Economy']
for folder in folders:
    category_folder_path = os.path.join(folder_path, folder)
    data.extend(load_data_from_txt(category_folder_path, folder))

# Create DataFrame
df = pd.DataFrame(data, columns=['text', 'category'])

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(classifier, 'model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate the classifier
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save the classification report to a file
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))

print(classification_report(y_test, y_pred))



