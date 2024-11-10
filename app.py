
import streamlit as st
import numpy as np
import re
import joblib  # or any other library for loading your model

# Load your model
@st.cache_resource
def load_model():
    # Replace with your model loading code
    model = joblib.load('/content/\content\model.joblib')  # Adjust if you exported from the notebook
    return model

model = load_model()
max_desc_length=200

vocabulary=set() #unique list of all words from all description

def add_to_vocab(df, vocabulary):
    for i in df.clean_desc:
        for word in i.split():
            vocabulary.add(word)
    return vocabulary

vocabulary=add_to_vocab(clean_book, vocabulary)

#This dictionary represents the mapping from word to token. Using token+1 to skip 0, since 0 will be used for padding descriptions with less than 200 words
vocab_dict={word: token+1 for token, word in enumerate(list(vocabulary))}

#This dictionary represents the mapping from token to word
token_dict={token+1: word for token, word in enumerate(list(vocabulary))}

assert token_dict[1]==token_dict[vocab_dict[token_dict[1]]]

def tokenizer(desc, vocab_dict, max_desc_length):
    '''
    Function to tokenize descriptions
    Inputs:
    - desc, description
    - vocab_dict, dictionary mapping words to their corresponding tokens
    - max_desc_length, used for pre-padding the descriptions where the no. of words is less than this number
    Returns:
    List of length max_desc_length, pre-padded with zeroes if the desc length was less than max_desc_length
    '''
    a=[vocab_dict[i] if i in vocab_dict else 0 for i in desc.split()]
    b=[0] * max_desc_length
    if len(a)<max_desc_length:
        return np.asarray(b[:max_desc_length-len(a)]+a).squeeze()
    else:
        return np.asarray(a[:max_desc_length]).squeeze()

def _removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"'s", " is ", text)
    text = re.sub(r"'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"'re", " are ", text)
    text = re.sub(r"'d", " would ", text)
    text = re.sub(r"'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text
def reviewBook(model,text):
    labels = ['fiction', 'nonfiction']
    a = clean_text(sentence)
    a = tokenizer(a, vocab_dict, max_desc_length)
    a = np.reshape(a, (1,max_desc_length))
    output = model.predict(a, batch_size=1)
    score = (output>0.5)*1
    pred = score.item()
    return labels[pred]

def predict_genre(sentence):
    # Replace with your prediction function
    prediction = reviewBook(model,sentence)  # Adjust based on your model
    return prediction

# Streamlit app
st.title("Book Genre Prediction")

st.write("Type a sentence and get a prediction of the book genre.")

sentence = st.text_input("Enter a sentence from a book:")

if st.button("Predict Genre"):
    if sentence:
        genre = predict_genre(sentence)
        st.write(f"The predicted genre is: **{genre}**")
    else:
        st.write("Please enter a sentence.")
