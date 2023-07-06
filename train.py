# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import f1_score
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# import numpy as np

# # Load data
# data = pd.read_csv('data.csv')

# # Preprocess text data
# # ...

# # Split data into training and validation sets
# train_data = data.sample(frac=0.8, random_state=42)
# val_data = data.drop(train_data.index)

# # Convert text data to a matrix of TF-IDF features
# vectorizer = TfidfVectorizer()
# train_matrix = vectorizer.fit_transform(train_data['text'])
# val_matrix = vectorizer.transform(val_data['text'])

# # Train cosine similarity model
# thresholds = np.arange(0.1, 1.0, 0.1)
# max_f1 = 0
# best_threshold = 0
# for threshold in thresholds:
#     similarity_scores = cosine_similarity(val_matrix, train_matrix)
#     predicted_labels = []
#     for scores in similarity_scores:
#         max_score = max(scores)
#         if max_score > threshold:
#             predicted_labels.append(train_data.loc[scores == max_score, 'label'].values[0])
#         else:
#             predicted_labels.append('unknown')
#     f1 = f1_score(val_data['label'], predicted_labels, average='weighted')
#     if f1 > max_f1:
#         max_f1 = f1
#         best_threshold = threshold

# # Print best threshold and F1 score
# print('Best threshold: {:.2f}'.format(best_threshold))
# print('Best F1 score: {:.2f}'.format(max_f1))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import requests
import json
import string
from math import sqrt
from collections import Counter
from itertools import chain
from functools import reduce
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Read text data from file
f = open('test1.txt', 'r')
dataa=f.read()


# Preprocess text data
# ...
def pre_process(answer):
    # text=answer.read().decode()
    # Case folding
    text=answer
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Removing punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.translate(table) for token in tokens]
    stripped = [token for token in stripped if token.isalpha() or token.isdigit()]
    
    return stripped
data=pre_process(dataa)

# Split data into training and validation sets
train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):]

# Convert text data to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
train_matrix = vectorizer.fit_transform(train_data)
val_matrix = vectorizer.transform(val_data)

# Train cosine similarity model
thresholds = np.arange(0.1, 1.0, 0.1)
max_f1 = 0
best_threshold = 0
for threshold in thresholds:
    similarity_scores = cosine_similarity(val_matrix, train_matrix)
    predicted_labels = []
    for scores in similarity_scores:
        max_score = max(scores)
        if max_score > threshold:
            predicted_labels.append(train_data[scores == max_score][0].split()[0])
        else:
            predicted_labels.append('unknown')
    f1 = f1_score([d.split()[0] for d in val_data], predicted_labels, average='weighted')
    if f1 > max_f1:
        max_f1 = f1
        best_threshold = threshold

# Print best threshold and F1 score
print('Best threshold: {:.2f}'.format(best_threshold))
print('Best F1 score: {:.2f}'.format(max_f1))
