import spacy

nlp = spacy.load('en_core_web_md')

# Function to calculate the Jaccard similarity between two documents
def jaccard_similarity(file1, file2):
    # Open and read the first file
    with open(file1, 'r') as f1:
        doc1 = nlp(f1.read())
    # Open and read the second file
    with open(file2, 'r') as f2:
        doc2 = nlp(f2.read())
    # Get the set of unique tokens in each document
    tokens1 = set([token.text for token in doc1 if not token.is_punct and not token.is_stop])
    tokens2 = set([token.text for token in doc2 if not token.is_punct and not token.is_stop])
    # Calculate the Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    similarity = intersection / union
    return similarity

# Example usage
file1 = 'test1.txt'
file2 = 'test2.txt'
similarity = jaccard_similarity(file1, file2)
print("Jaccard similarity between the two files: ", similarity)
