import streamlit as st
import torch
import spacy

from SkimlitData import SkimlitDataset
from WordEmbeddings import get_embeddings
from SkimlitClassifier import SkimlitModel
from Tokenizer import Tokenizer
from LabelEncoder import LabelEncoder
from MakePredictions import make_skimlit_predictions, example_input

MODEL_PATH = 'utils/skimlit-model-final-1.pt'
TOKENIZER_PATH = 'utils/tokenizer.json'
LABEL_ENCODER_PATH = "utils/label_encoder.json"
EMBEDDING_FILE_PATH = 'utils/glove.6B.300d.txt'

@st.cache_data()
def create_utils(model_path, tokenizer_path, label_encoder_path, embedding_file_path):
    tokenizer = Tokenizer.load(fp=tokenizer_path)
    label_encoder = LabelEncoder.load(fp=label_encoder_path)
    embedding_matrix = get_embeddings(embedding_file_path, tokenizer, 300)
    model = SkimlitModel(embedding_dim=300, vocab_size=len(tokenizer), hidden_dim=128, n_layers=3, linear_output=128, num_classes=len(label_encoder), pretrained_embeddings=embedding_matrix)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(model)
    return model, tokenizer, label_encoder

def model_prediction(abstract, model, tokenizer, label_encoder):
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    lines, pred = make_skimlit_predictions(abstract, model, tokenizer, label_encoder)
    # pred, lines = make_predictions(abstract)

    for i, line in enumerate(lines):
        if pred[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif pred[i] == 'BACKGROUND':
            background = background + line
        
        elif pred[i] == 'METHODS':
            method = method + line
        
        elif pred[i] == 'RESULTS':
            result = result + line
        
        elif pred[i] == 'CONCLUSIONS':
            conclusion = conclusion + line

    return objective, background, method, conclusion, result



def main():
    st.set_page_config(
        page_title="SkimLit",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("<h1 style='text-align: center; color: LightGray;'>Skimlit📄⚡</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'> Find the information you need in abstracts faster than ever. This NLP-powered app automatically classifies each sentence into a relevant heading, so you can quickly skim through abstracts and find the information you need.", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0; margin-bottom: 0;'>", unsafe_allow_html=True)

    
    # Creating model, tokenizer, and label encoder
    skimlit_model, tokenizer, label_encoder = create_utils(MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, EMBEDDING_FILE_PATH)

    col1, col2 = st.columns(2)

    with col1:
        abstract = st.text_area(label='**Enter Abstract Here!!**', height=120)

        agree = st.checkbox('Show Example Input')
        if agree:
            st.info(example_input)


        predict = st.button('Extract!')

    # Make prediction button logic
    if predict:
        with st.spinner('Wait for prediction....'):
            objective, background, methods, conclusion, result = model_prediction(abstract, skimlit_model, tokenizer, label_encoder)

        with col2:
            st.markdown(f'### Objective:')
            st.write(f'{objective}')
            st.markdown(f'### Background:')
            st.write(f'{background}')
            st.markdown(f'### Methods:')
            st.write(f'{methods}')
            st.markdown(f'### Result:')
            st.write(f'{result}')
            st.markdown(f'### Conclusion:')
            st.write(f'{conclusion}')

          

if __name__ == '__main__':
    main()