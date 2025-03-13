# %%
import streamlit as st
import sys
import os
import subprocess

# Tampilkan informasi sistem
st.write(f"Python version: {sys.version}")
st.write(f"Current directory: {os.getcwd()}")

# Coba impor dan instalasi paket yang diperlukan
required_packages = ['nltk', 'scikit-learn', 'numpy']

for package in required_packages:
    try:
        if package == 'scikit-learn':
            import sklearn
            st.success(f"scikit-learn sudah terinstal!")
        elif package == 'nltk':
            import nltk
            st.success(f"NLTK sudah terinstal!")
        elif package == 'numpy':
            import numpy
            st.success(f"NumPy sudah terinstal!")
    except ImportError:
        st.warning(f"{package} tidak ditemukan. Mencoba menginstal...")
        try:
            package_name = package
            if package == 'scikit-learn':
                package_name = 'scikit-learn'
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            
            if package == 'scikit-learn':
                import sklearn
            elif package == 'nltk':
                import nltk
            elif package == 'numpy':
                import numpy
                
            st.success(f"{package} berhasil diinstal!")
        except Exception as e:
            st.error(f"Gagal menginstal {package}: {e}")
            st.stop()

# Setelah NLTK berhasil diimpor, unduh data yang diperlukan
if 'nltk' in sys.modules:
    try:
        # Buat direktori untuk data jika belum ada
        if not os.path.exists("nltk_data"):
            os.makedirs("nltk_data")
        
        # Atur path data NLTK
        nltk.data.path.append(os.path.abspath("nltk_data"))
        
        # Unduh data punkt
        st.write('Mengunduh data NLTK punkt...')
        nltk.download('punkt', download_dir='nltk_data')
        st.success("Data NLTK punkt berhasil diunduh!")
    except Exception as e:
        st.error(f"Gagal mengunduh data NLTK: {e}")

# Sekarang coba impor komponen spesifik dari scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    st.success("Berhasil mengimpor komponen scikit-learn!")
except Exception as e:
    st.error(f"Error saat mengimpor komponen scikit-learn: {e}")
    st.stop()

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


