import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
# Load the saved model
model = joblib.load('model.pkl')

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load('vectorizer.pkl')


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

# Function to predict the category of a new text input
def predict_category(text,threshold=0.5):
    # Preprocess the text input
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using the TF-IDF vectorizer
    text_vectorized = vectorizer.transform([preprocessed_text])
    
    # Use the trained model to predict the probabilities for each category
    predicted_probs = model.predict_proba(text_vectorized)[0]
    
    # Get the maximum probability and corresponding category
    max_prob = max(predicted_probs)
    predicted_category = model.classes_[predicted_probs.argmax()]
    
    # Check if the maximum probability is above the threshold
    if max_prob < threshold:
      predicted_category="artical is out of the model categories"
    
    return predicted_category

# Read text from file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example usage
file_path = 'Articel txt file path here'
text_input = read_text_from_file(file_path)
predicted_category = predict_category(text_input)
print("Predicted category:", predicted_category)