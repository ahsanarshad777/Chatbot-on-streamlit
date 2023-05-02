import streamlit as st
from streamlit_chat import message as st_message

import numpy as np
import json
from tensorflow import keras
import pickle


@st.experimental_singleton
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    with open('intents.json', encoding="utf8") as file:
        data = json.load(file)

    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return data, tokenizer, model, lbl_encoder


if "history" not in st.session_state:
    st.session_state.intro = ''

st.title("Hello Chatbot")


def generate_answer():

    # parameters
    max_len = 20

    data, tokenizer, model, lbl_encoder = get_models()
    user_message = st.session_state.input_text

    result = model.predict(keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_message]), truncating='post', maxlen=max_len))

    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            st.session_state.intro = np.random.choice(i['responses'])


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)


st_message(**st.session_state.intro)  # unpacking
