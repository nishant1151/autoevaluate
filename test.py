from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import pandas as pd

def accuracy():
# Sample training data
        train_data = pd.DataFrame({
        'text': ['The quick brown fox jumps over the lazy dog', 
                'A quick brown dog jumps over the lazy fox',
                'The sky is blue',
                'The grass is green'],
        'label': ['animal', 'animal', 'color', 'color']
        })

        # Sample test data
        test_data = pd.DataFrame({
        'text': ['The quick brown fox', 'The sky is blue'],
        'label': ['animal', 'color']
        })

        # Convert training data to a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        train_matrix = vectorizer.fit_transform(train_data['text'])

        # Calculate cosine similarity between test and training data
        test_matrix = vectorizer.transform(test_data['text'])
        similarity_scores = cosine_similarity(test_matrix, train_matrix)

        # Predict labels based on similarity scores and a threshold
        threshold = 0.5
        predicted_labels = []
        for scores in similarity_scores:
                max_score = max(scores)
                if max_score > threshold:
                        predicted_labels.append(train_data.loc[scores == max_score, 'label'].values[0])
                else:
                        predicted_labels.append('unknown')

        # Calculate F1 score
        f1 = f1_score(test_data['label'], predicted_labels, average='weighted')
        print('Accuracy: {:.2f}'.format(f1*90))

accuracy()