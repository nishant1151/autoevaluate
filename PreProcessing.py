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
# from c.models.keyedvectors import KeyedVectors 
import math
from werkzeug.wrappers import Request, Response
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from gensim.similarities import Similarity
import result
import flask
import sqlite3
from flask import Flask, request, render_template,session, redirect, url_for



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
    # if 'logged_in' in session:
    #     return stripped
    # else:
    #     return redirect(url_for('login'))


def check_grammar(text,total_marks):
    marks=(int(total_marks)*10)/100
    url = 'https://languagetool.org/api/v2/check'
    data = {
        'text': text,
        'language': 'en-US'
    }
    response = requests.post(url, data=data)
    result = response.json()
    errors = result['matches']
    no_of_errors = len(errors)
    if no_of_errors>100:
        g=0*marks
    elif no_of_errors>=90 and no_of_errors<=100:
        g=0.1*marks
    elif no_of_errors>=80 and no_of_errors<90:
        g=0.2*marks
    elif no_of_errors>=70 and no_of_errors<80:
        g=0.3*marks
    elif no_of_errors>=60 and no_of_errors<70:
        g=0.4*marks
    elif no_of_errors>=50 and no_of_errors<60:
        g=0.5*marks
    elif no_of_errors>=40 and no_of_errors<50:
        g=0.6*marks
    elif no_of_errors>=30 and no_of_errors<40:
        g=0.7*marks
    elif no_of_errors>=20 and no_of_errors<30:
        g=0.8*marks
    elif no_of_errors>=10 and no_of_errors<20:
        g=0.9*marks
    else:
        g=1*marks
        
    return g
    if 'logged_in' in session:
        return str(g)
    else:
        return redirect(url_for('login'))


#length of string
def CheckLenght(client_answer,total_marks):
    marks=(int(total_marks)*10)/100
    client_ans = len(client_answer.split()) 
    #return client_ans
    kval1 = 0
    if client_ans > 100:
        kval1 = 0.9*marks
    elif client_ans > 75:
        kval1 = 0.7*marks
    elif client_ans > 50:
        kval1 = 0.5*marks
    elif client_ans > 25:
        kval1 = 0.4*marks
    elif client_ans > 10:
        kval1 = 0.2*marks
    else:
        kval1 = 0.1*marks
    return kval1
   





def cosine__similarity(words1, words2,total_marks):
    marks=(int(total_marks)*20)/100
    word_set = set(words1).union(set(words2))
    word_dict1 = Counter(words1)
    word_dict2 = Counter(words2)
    numerator = reduce(lambda x, y: x + y, map(lambda w: word_dict1[w] * word_dict2[w], word_set))
    denominator1 = sqrt(reduce(lambda x, y: x + y, map(lambda w: word_dict1[w]**2, word_dict1.keys())))
    denominator2 = sqrt(reduce(lambda x, y: x + y, map(lambda w: word_dict2[w]**2, word_dict2.keys())))
    denominator = denominator1 * denominator2
    if not denominator:
        return 0.0
    else:
        return (float(numerator) / denominator)*marks
    
# contraduction

def contradiction(text1,text2,total_marks):
    marks=(int(total_marks)*10)/100

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased')



    # Encode the two texts using the tokenizer and pass the resulting tensors to the model
    encoded_text1 = tokenizer(text1, return_tensors='pt')
    encoded_text2 = tokenizer(text2, return_tensors='pt')

    # Get the model outputs for the encoded texts
    outputs1 = model(**encoded_text1)
    outputs2 = model(**encoded_text2)

    # Compare the encoded vectors of the two texts using cosine similarity
    from torch.nn import CosineSimilarity
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos_sim(outputs1.last_hidden_state.mean(dim=1), outputs2.last_hidden_state.mean(dim=1))

    if similarity < 0.8:
        return marks
    elif similarity <0.5:
        return marks/5
    else:
        return 0
    




def sementic_similarity(text1,text2,total_marks):
    marks=(int(total_marks)*20)/100
    # Tokenize and preprocess texts
    text1_tokens = text1
    text2_tokens = text2

    # Combine tokens into sentences
    text1_sentences = [' '.join(text1_tokens)]
    text2_sentences = [' '.join(text2_tokens)]

    # Create TfidfVectorizer instance
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(text1_sentences + text2_sentences)

    # Extract the feature vectors for the two texts
    text1_vector = tfidf_matrix[0]
    text2_vector = tfidf_matrix[1]

    # Calculate cosine similarity
    similarity_score = cosine_similarity(text1_vector, text2_vector)
    return (similarity_score[0][0])*marks

    
def calculate_conceptual_similarity(text1,text2,total_marks):
    marks=(int(total_marks)*20)/100
    
    # tokenize and create a dictionary of words
    texts = [text1, text2]
    dictionary = corpora.Dictionary(texts)

    # convert texts into vectors using Bag of Words model
    corpus = [dictionary.doc2bow(text) for text in texts]

    # train the LSI model on the corpus
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    # convert the texts into LSI vectors
    text1_lsi = lsi[dictionary.doc2bow(text1)]
    text2_lsi = lsi[dictionary.doc2bow(text2)]

    # compute similarity score using cosine similarity of LSI vectors
    similarity_score = Similarity('', corpus=lsi[corpus], num_features=2)[text1_lsi][1]
    return (similarity_score*marks)
    







def KeyWordmatching(text1, text2,total_marks):
    marks=(int(total_marks)*20)/100
    # Create a CountVectorizer object to tokenize the text
    vectorizer = CountVectorizer()

    # Fit the CountVectorizer object to the text and transform the text into a document-term matrix
    dtm = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity between the two rows of the document-term matrix
    cosine_sim = cosine_similarity(dtm)[0][1]

    # Calculate the number of matching keywords as the cosine similarity multiplied by the total number of unique words
    keywords = round(cosine_sim * len(set(text1.split() + text2.split())))
    kval1=0
    if keywords > 100:
        kval1 = 0.9*marks
    elif  keywords> 75:
        kval1 = 0.7*marks
    elif keywords> 50:
        kval1 = 0.5*marks
    elif  keywords> 25:
        kval1 = 0.4*marks
    elif keywords > 10:
        kval1 = 0.2*marks
    else:
        kval1 = 0.1*marks
   

    return kval1



def main(answer,answer_key,total_marks,std_name,std_email,sub):
    # source_doc1=answer
    # target_docs=answer_key
    key_length=CheckLenght(answer,total_marks)
    key_Error=check_grammar(answer,total_marks)
    pre_proce_answer=pre_process(answer)
    pre_proce_answer_key=pre_process(answer_key)
    key_match=cosine__similarity(pre_proce_answer,pre_proce_answer_key,total_marks) 
    # answer_contradiction=contradiction(answer,answer_key,total_marks)
    answer_sementic_similarity=sementic_similarity(pre_proce_answer,pre_proce_answer_key,total_marks)
    answer_conceptual_simililarity=calculate_conceptual_similarity(pre_proce_answer,pre_proce_answer_key,total_marks)




    answer_keyword=KeyWordmatching(answer,answer_key,total_marks)
    final_marks=0
    if int(answer_sementic_similarity)==0 or int(key_match) == 0  :
        
        final_marks=0
        if 'logged_in' in session:
            return result.result(final_marks,std_name,total_marks,sub,std_email)
        else:
            return redirect(url_for('login'))
    else:
        final_marks=(key_length+key_Error+key_match+answer_conceptual_simililarity+answer_sementic_similarity+answer_keyword)
    # return str(int(final_marks))
        if 'logged_in' in session:
            return result.result(final_marks,std_name,total_marks,sub,std_email)
        else:
            return redirect(url_for('login'))

    


