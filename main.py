import numpy as np
import tensorflow as tf

# Imdb dataset available in Tensor flow itselt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN , Dense
from tensorflow.keras.models import load_model

#Loading IMDB Dataset word index

word_index=imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#Step -1        Load the pre-trained model with RELU activation
model = load_model('simple_rnn_imdb.h5')

## Step -2 --->     Helper Functions

# Function to Decode reviews
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to Process user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##step --3   prediction function

## Prediction Function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment="Audience is positive towards Movie" if prediction[0][0] > 0.5 else "Audience is Negative towards Movie"

    return sentiment, prediction[0][0]


##***********************************************************************************************************************************
## ----------------------------------  DESIGNING STREAMLIT APP ---------------------------------------------------------------------
##***********************************************************************************************************************************

import streamlit as st

st.title("IMDB Movie Review (sentiment of Audience)")

st.write("Enter your Review to classify it is a positive or negative review")

# user Input
user_input = st.text_area("Movie Review")

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    # Making Prediction
    prediction = model.predict(preprocessed_input)
    sentiment="Audience is positive towards Movie" if prediction[0][0] > 0.5 else "Audience is Negative towards Movie"



    # Display the result

    st.write(f'Sentiment is : {sentiment}')
    st.write(f'Prediction Score is : {prediction[0][0]}')

else:
    st.write("Please enter review")


