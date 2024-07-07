import numpy as np
import spacy
import nltk
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from warnings import filterwarnings
filterwarnings('ignore')



def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Document Classification', layout='centered')

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Financial Document Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


def text_extract_from_html(html_file):

    # Read the uploaded HTML file
    html_content = html_file.read().decode('utf-8')

    # Parse the HTML Content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the Text
    text = soup.get_text()

    # Split the Text and Remove Unwanted Space
    result = [i.strip() for i in text.split()]

    return result


def text_processing(text):

    # spaCy Engine
    nlp = spacy.load('en_core_web_lg')

    # Process the Text with spaCy
    doc = nlp(' '.join(text))

    # Tokenization, Lemmatization, and Remove Stopwords, punctuation, digits
    token_list = [
                  token.lemma_.lower().strip()
                  for token in doc
                  if token.text.lower() not in nlp.Defaults.stop_words and token.text.isalpha()
                 ]

    if len(token_list) > 0:
        return ' '.join(token_list)
    else:
        return 'empty'
    

def sentence_embeddings(sentence):

    # split the sentence into separate words
    words = word_tokenize(sentence)                         

    # load the trained model
    model = Word2Vec.load('word2vec_model.bin')

    # get the vectors of each words
    vectors = [model.wv[word] for word in words if word in model.wv]
    
    if vectors:
        # return the average of vectors
        return np.mean(vectors, axis=0)

    else:
        # we set the model parameter in training ---> vector_size = 300
        return np.zeros(model.vector_size)


def prediction(input_file):

    # Extract the Text from HTML Document
    html_content = text_extract_from_html(input_file)

    # Preprocess the Text
    preprocessed_text = text_processing(html_content)

    # Text Convert into Embeddings
    features = sentence_embeddings(preprocessed_text)

    # Reshape the features into match the expected input shape of Model
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    # Convert into Tensors
    features_tensors = tf.convert_to_tensor(features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('model.h5', custom_objects = {'Orthogonal': tf.keras.initializers.Orthogonal})
    prediction = model.predict(features_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name
    target = {0:'Balance Sheets', 1:'Cash Flow', 2:'Income Statement', 3:'Notes', 4:'Others'}
    predicted_class = target[target_label]

    # Find the Confidence
    confidence = round(np.max(prediction)*100, 2)

    add_vertical_space(2)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>', 
                    unsafe_allow_html=True)
    
    add_vertical_space(1)
    st.markdown(f'<h3 style="text-align: center; color: green;">Prdicted Class = {predicted_class}</h3>', 
                    unsafe_allow_html=True)



# Streamlit Configuration Setup
streamlit_config()
    

# File uploader to upload the HTML file
input_file = st.file_uploader('Upload an HTML file', type='html')

if input_file is not None:

    try:
        # Predict the Input_HTML_File_Class
        prediction(input_file)

    except:
        # Check 'punkt' Already Downloaded or Not
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Predict the Input_HTML_File_Class
        prediction(input_file)
