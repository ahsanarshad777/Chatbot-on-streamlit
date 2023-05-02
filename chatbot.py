import numpy as np
import json
import playsound
import datetime
import speech_recognition as sr
from gtts import gTTS
from tensorflow import keras
import pickle

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


class ChatBot():

    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name
        self.text = ""

    """Speech Recognition"""

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio, language="en-UK")
            print("me --> ", self.text)
        except:
            print("me -->  ERROR")

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        filename = "voice"+date_string+".mp3"
        speaker.save(filename)
        print("audio file saved as ", filename)
        playsound.playsound(filename)

    def wake_up(self, text):
        return True if self.name in text else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


"""Run the AI"""

# Run the AI
if __name__ == "__main__":
    ai = ChatBot(name="Siri")

    while True:
        ai.speech_to_text()

        # parameters
        max_len = 20

        inp = ai.text

        # wake up
        if ai.wake_up(ai.text) is True:
            intro = "Asssalamu alaikum، I am Siri، an AI Chatbot, how can i help you?"

        # action time
        elif "time" in ai.text:
            intro = ai.action_time()
        # respond politely
        elif any(i in ai.text for i in ["thanks", "Thank you"]):
            intro = np.random.choice(["happy to help you.", "i am here  if you need more assistance",
                                     "mention not", "if you need more help let me know.", "it is my duty to serve you. "])

        # conversation
        else:
            result = model.predict(keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])

            for i in data['intents']:
                if i['tag'] == tag:
                    intro = np.random.choice(i['responses'])
                    previous_intro = intro
            # if i not in data['intents']:
            #     intro = "I'm sorry, I didn't understand what you said. Please try again."
    
            ai.text_to_speech(intro)
        