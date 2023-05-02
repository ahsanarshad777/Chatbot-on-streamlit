import streamlit as st
import numpy as np
import json
import datetime
import speech_recognition as sr
from gtts import gTTS
from tensorflow import keras
import pickle
import playsound

# Load the intents.json file
with open('intents.json', encoding="utf8") as file:
    data = json.load(file)

# Load the trained model
model = keras.models.load_model('chat_model')

# Load the tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Define a function to convert text to speech using gTTS


def text_to_speech(text):
    speaker = gTTS(text=text, lang="en", slow=False)
    date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice"+date_string+".mp3"
    speaker.save(filename)
    playsound.playsound(filename)

# Define a function to get the bot's response


def get_bot_response(user_input):
    # Set the maximum input length
    max_len = 20
    # Convert the user's input to a padded sequence of integers using the tokenizer
    input_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len)
    # Use the model to predict the appropriate tag for the user's input
    result = model.predict(input_seq)
    # Convert the predicted tag back to text using the label encoder
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    # Choose a random response from the appropriate intent in the intents.json file
    for i in data['intents']:
        if i['tag'] == tag:
            bot_response = np.random.choice(i['responses'])
    # Return the bot's response
    return bot_response

# Define a function to run the chatbot


def run_chatbot():
    # Set the bot's name
    bot_name = "Siri"
    st.title(
        f"{bot_name}: Asssalamu Alaikum، I am Siri، an AI Chatbot, how can i help you?")
    st.markdown("Lets Chat!")
    # Create a loop to run the chatbot
    user_input = st.text_input('You:', '')
    bot_response = get_bot_response(user_input)
    st.text_area("Siri", value=bot_response,
                 height=100, max_chars=None)
    text_to_speech(bot_response)
    # st.audio("voice*.mp3")


run_chatbot()
