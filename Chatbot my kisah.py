# %%
import os 
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("My Kisah")

# Tampilkan informasi diagnostik
st.write(f"Python version: {sys.version}")
st.write("Mencoba menginstal dan memuat NLTK...")

# Coba instal NLTK via subprocess jika tidak tersedia
try:
    import nltk
except ImportError:
    st.warning("NLTK tidak terdeteksi, mencoba menginstal...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
        st.success("NLTK berhasil diinstal melalui pip!")
    except:
        st.error("Gagal menginstal NLTK melalui pip.")

# Coba download data NLTK
try:
    if 'nltk' in sys.modules:
        # Buat direktori nltk_data jika belum ada
        if not os.path.exists("./nltk_data"):
            os.makedirs("./nltk_data")
        
        # Set path NLTK data
        nltk.data.path.append(os.path.abspath("./nltk_data"))
        
        # Download punkt dengan menampilkan proses
        st.write("Downloading NLTK punkt...")
        nltk.download('punkt', download_dir='./nltk_data')
        st.success("NLTK punkt berhasil didownload!")
except Exception as e:
    st.error(f"Error saat download NLTK data: {str(e)}")


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


