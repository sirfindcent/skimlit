# Citing code from VishalRK1 (2022):
# MakePredictions.py.
# (Version 1.0) [Source code].
# https://github.com/vishalrk1/SkimLit/blob/main/MakePredictions.py

import numpy as np
import spacy
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import torch
import torch.nn.functional as F

from SkimlitData import SkimlitDataset

def download_stopwords():
    nltk.download("stopwords")
    STOPWORDS = stopwords.words("english")
    porter = PorterStemmer()
    return STOPWORDS, porter

def preprocess(text, stopwords):
    """Conditional preprocessing on our text unique to our task."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Remove words in paranthesis
    text = re.sub(r"\([^)]*\)", "", text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    return text

def spacy_function(abstract):
  nlp = spacy.load("en_core_web_sm") # setup english sentence parser 
  doc = nlp(abstract) # create "doc" of parsed sequences
  abstract_lines = [str(sent) for sent in list(doc.sents)] # list of line on string (not spaCy type)
    
  return abstract_lines
    
# ---------------------------------------------------------------------------------------------------------------------------

def model_prediction(model, dataloader):
  """Prediction step."""
  # Set model to eval mode
  model.eval()
  y_trues, y_probs = [], []
  # Iterate over val batches
  for i, batch in enumerate(dataloader):
    # Forward pass w/ inputs
    # batch = [item.to(.device) for item in batch]  # Set device
    inputs = batch
    z = model(inputs)
    # Store outputs
    y_prob = F.softmax(z, dim=1).detach().cpu().numpy()
    y_probs.extend(y_prob)
  return np.vstack(y_probs)

# ---------------------------------------------------------------------------------------------------------------------------

def make_skimlit_predictions(text, model, tokenizer, label_encoder): # embedding path
  # getting all lines separated from abstract
  abstract_lines = list()
  abstract_lines = spacy_function(text)  
    
  # Get total number of lines
  total_lines_in_sample = len(abstract_lines)

  # Go through each line in abstract and create a list of dictionaries containing features for each line
  sample_lines = []
  for i, line in enumerate(abstract_lines):
    sample_dict = {}
    sample_dict["text"] = str(line)
    sample_dict["line_number"] = i
    sample_dict["total_lines"] = total_lines_in_sample - 1
    sample_lines.append(sample_dict)

  # converting sample line list into pandas Dataframe
  df = pd.DataFrame(sample_lines)
  
  # getting stopword
  STOPWORDS, porter = download_stopwords()

  # applying preprocessing function to lines
  df.text = df.text.apply(lambda x: preprocess(x, STOPWORDS))

  # converting texts into numberical sequences
  text_seq = tokenizer.texts_to_sequences(texts=df['text'])

  # creating Dataset
  dataset = SkimlitDataset(text_seq=text_seq, line_num=df['line_number'], total_line=df['total_lines'])

  # creating dataloader
  dataloader = dataset.create_dataloader(batch_size=2)

  # Preparing embedings
#   embedding_matrix = get_embeddings(embeding_path, tokenizer, 300)

  # creating model
#   model = SkimlitModel(embedding_dim=300, vocab_size=len(tokenizer), hidden_dim=128, n_layers=3, linear_output=128, num_classes=len(label_encoder), pretrained_embeddings=embedding_matrix)

  # loading model weight
#   model.load_state_dict(torch.load('/content/drive/MyDrive/Datasets/SkimLit/skimlit-pytorch-1/skimlit-model-final-1.pt', map_location='cpu'))

  # setting model into evaluation mode
  model.eval()

  # getting predictions 
  y_pred = model_prediction(model, dataloader)

  # converting predictions into label class
  pred = y_pred.argmax(axis=1)
  pred = label_encoder.decode(pred)

  return abstract_lines, pred

example_input = '''
The aim of this study was to assess the endorsement of reporting guidelines in Korean traditional medicine (TM) journals by reviewing their instructions to authors. We examined the instructions to authors in all of the TM journals published in Korea to assess the appropriate use of reporting guidelines for research studies. The randomized controlled trials (RCTs) published after 2010 in journals that endorsed reporting guidelines were obtained. The reporting quality was assessed using the following guidelines: the 38-item Consolidated Standards of Reporting Trials (CONSORT) statement for non-pharmacological trials (NPT); the 17-item Standards for Reporting Interventions in Clinical Trials of Acupuncture (STRICTA) statement, instead of the 5-item CONSORT for acupuncture trials; and the 22-item CONSORT extensions for herbal medicine trials. The overall item score was calculated and expressed as a proportion.One journal that endorsed reporting guidelines was identified. Twenty-nine RCTs published in this journal after 2010 met the selection criteria. General editorial policies such as those of the International Committee of Medical Journal Editors (ICMJE) were endorsed by 15 journals. In each of the CONSORT-NPT articles, 21.6 to 56.8% of the items were reported, with an average of 11.3 items (29.7%) being reported. In the 24 RCTs (24/29, 82.8%) appraised using the STRICTA items, an average of 10.6 items (62.5%) were addressed, with a range of 41.2 to 100%. For the herbal intervention reporting, 17 items (77.27%) were reported. In the RCT studies before and after the endorsement of CONSORT and STRICTA guidelines by each journal, all of the STRICTA items had significant improvement, whereas the CONSORT-NPT items improved without statistical significance.The endorsement of reporting guidelines is limited in the TM journals in Korea. Authors should adhere to the reporting guidelines, and editorial departments should refer authors to the various reporting guidelines to improve the quality of their articles.
'''
