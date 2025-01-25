import nltk
import random
import json
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from intents.json
with open("intents.json") as file:
    intents = json.load(file)

# Prepare the data for training
training_sentences = []
training_labels = []

# Loop through each intent and extract the patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

# Lemmatize each word in the training sentences
def preprocess_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return " ".join(words)

training_sentences = [preprocess_sentence(sentence) for sentence in training_sentences]

# Convert labels into numeric values
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Train the SVM model using a pipeline that combines CountVectorizer and LinearSVC
model = make_pipeline(CountVectorizer(), LinearSVC())
model.fit(training_sentences, training_labels)

# Context variable to track conversation
current_context = None

# Function to predict intent for a given sentence
def predict_class(sentence):
    # Preprocess the sentence and predict the class
    processed_sentence = preprocess_sentence(sentence)
    prediction = model.predict([processed_sentence])
    predicted_intent = label_encoder.inverse_transform(prediction)
    return predicted_intent[0]

# Function to get a response based on the predicted intent and context
def get_response(predicted_intent, context):
    global current_context
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            # Check if the context matches
            if "context_filter" in intent and intent["context_filter"] != context:
                return "I'm not sure I understand. Can you clarify?"
            # Update context if applicable
            if "context_set" in intent:
                current_context = intent["context_set"]
            return random.choice(intent['responses'])
    return "I'm not sure I understand."

# Initialize session state to keep track of conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit Interface
st.title("Farmer Assistant Chatbot ")

st.subheader("FarmAssist")
st.write("Talk to the bot!")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", options=["Chat", "Conversation History"])

if menu == "Chat":
    # User Input and Chatbot Response
    user_input = st.text_input("You: ", key="user_input")

    if user_input:
        if user_input.lower() == "quit":
            st.write("Chatbot: Goodbye!")
        else:
            predicted_intent = predict_class(user_input)
            response = get_response(predicted_intent, current_context)
            st.session_state.history.append(f"You: {user_input}")
            st.session_state.history.append(f"Chatbot: {response}")
            st.write(f"Chatbot: {response}")

elif menu == "Conversation History":
    # Conversation History display
    st.subheader("Conversation History")
    conversation_history = "\n".join(st.session_state.history)
    st.text_area("History", value=conversation_history, height=400, key="history_area", disabled=True)


