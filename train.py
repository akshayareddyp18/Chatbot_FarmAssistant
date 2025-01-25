import json
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import joblib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents.json file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Data preprocessing
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

for intent in intents["intents"]:
    # Check if 'patterns' key exists in the intent
    if "patterns" in intent:
        for pattern in intent["patterns"]:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add to documents
            documents.append((pattern, intent["tag"]))  # Store the full sentence for TF-IDF
            # Add to classes
            if intent["tag"] not in classes:  # Use intent["tag"] here
                classes.append(intent["tag"])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Prepare the dataset
patterns = [doc[0] for doc in documents]  # Sentences
labels = [classes.index(doc[1]) for doc in documents]  # Numerical labels

# Debug: Check if patterns and labels are populated
print(f"Patterns: {patterns}")
print(f"Labels: {labels}")

# If the patterns or labels are empty, print an error and stop further processing
if len(patterns) == 0 or len(labels) == 0:
    raise ValueError("The dataset is empty. Please check the 'patterns' in your intents file.")

# Split data into training and testing
train_x, test_x, train_y, test_y = train_test_split(patterns, labels, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
train_x_tfidf = vectorizer.fit_transform(train_x)
test_x_tfidf = vectorizer.transform(test_x)

# Build and train the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')

svm_model.fit(train_x_tfidf, train_y)



# Save the model and supporting files
joblib.dump(svm_model, "model/intent_model_svm_rbf.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
np.save("model/classes.npy", classes)
np.save("model/words.npy", words)

print("SVM model trained and saved with TF-IDF and RBF kernel!")