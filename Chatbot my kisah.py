# %%
import os 
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


# %%
import json

with open('Chat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    intents = data['intents']  # Pastikan struktur file JSON benar

# %%
#membuat Vectorizer dan Classifier

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

#preprocess the data

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# %%
#Training the model
x= vectorizer.fit_transform(patterns)
y= tags
clf.fit(x,y)

# %%
#create chatbot

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break

# %%
#chatbot with streamlit

counter = 0
def main():
    global counter
    st.title("My Kisah")
    st.write("Hello, Darling. How can I help you today?")

    counter += 1
    user_input = st.text_input("You: ", key=f"user_input_{counter}")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("My Kisah: ", value=response, height=100, chars = none, key=f"My Kisah_response_{counter}")

        if response.lower() in ["bye", "goodbye", "see you"]:
            st.write("Oke, sayang")
            st.stop()

if __name__ == "__main__":
    main()


