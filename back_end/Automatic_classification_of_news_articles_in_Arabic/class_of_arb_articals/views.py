from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import joblib
from django.core.files.storage import FileSystemStorage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import easyocr






# Load the saved model
model = joblib.load('C:\\Users\\M-Tech\\Desktop\\Automatic classification of news articles in Arabic\\model.pkl')

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load('C:\\Users\\M-Tech\\Desktop\\Automatic classification of news articles in Arabic\\vectorizer.pkl')


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


# Function to extract text from image using OCR
def extract_text_from_image(image):
    # Open the image file
    reader = easyocr.Reader(['ar']) # 'ar' is for Arabic
    result = reader.readtext(image)
    text = ' '.join([item[1] for item in result])
    return text

# Create your views here.
# Endpoint for text input
@api_view(['POST'])
def classify_text(request):
    print(request.data)
    artical = request.data.get('article')
    artical = preprocess_text(artical)

    predicted_category = predict_category(artical)
    print("Predicted Category:", predicted_category)
    return Response({'categorie': predicted_category}, status=status.HTTP_200_OK)


# Endpoint for image input
@api_view(['POST'])
def classify_image(request):
    image = request.FILES.get('image')
    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    uploaded_file_url = fs.path(filename)
    print(uploaded_file_url)

    if not image:
        return Response({'error': 'No image uploaded'}, status=status.HTTP_400_BAD_REQUEST)
    
    artical = extract_text_from_image(uploaded_file_url)

    #arabic_regex = r'^[\u0600-\u06FF0-9\s,.:;"\'?!\/\\\-«»[\]()]*$'
    #if not re.match(arabic_regex, artical):
      #return Response({'error': 'Article contains non-Arabic characters'}, status=status.HTTP_400_BAD_REQUEST)
    
    artical_length = len(artical)
    print(artical_length )
    print(artical )
    if artical_length > 10000:
        return Response({'error': 'Artical exceeds the limit of 10000 characters'}, status=status.HTTP_400_BAD_REQUEST)
    elif artical_length < 300:
        return Response({'error': 'Artical must be at least 300 characters'}, status=status.HTTP_400_BAD_REQUEST)
    artical = preprocess_text(artical)
    predicted_category = predict_category(artical)
    print(predicted_category)
    
    return Response({'categorie': predicted_category}, status=status.HTTP_200_OK)