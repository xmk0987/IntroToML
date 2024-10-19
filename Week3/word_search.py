import random
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
""" print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(vocab['meeting'])
print(len(ivocab))
print(ivocab[10]) """

# W contains vectors for
""" print('Vocabulary word vectors') """
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
""" print(W.shape) """

def count_euclidean_distance(point1 ,point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point2[i] - point1[i]) ** 2
    return np.sqrt(distance)

def find_three_closest(input_word_index):
    input_vector = W[input_word_index]
    closest_words = {}
    
    for word, idx in vocab.items():
        new_word_vector = W[idx]
        distance = count_euclidean_distance(input_vector, new_word_vector)
        
        # Save the first three words
        if len(closest_words) <= 2:
            closest_words[word] = distance
            
        # Replace the word in dict with the biggest distance if new word distance shorter
        else:
            biggest_distance_word = max(closest_words, key=closest_words.get)
            biggest_distance = closest_words[biggest_distance_word]
            if biggest_distance > distance:
                del closest_words[biggest_distance_word]
                closest_words[word] = distance
            
    return closest_words
    
    
# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        # Find the index of the input word
        input_word_index = vocab[input_term]
        
        # Find the three closest words. Returned as dictionary        
        closest = find_three_closest(input_word_index)
        
        # Sort the dictionary based on the distance which is value in dict
        sorted_closest = {k: v for k, v in sorted(closest.items(), key=lambda item: item[1])}
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for word, distance in sorted_closest.items():
            print("%35s\t\t%f\n" % (word, distance))