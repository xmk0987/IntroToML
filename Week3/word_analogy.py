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

# Count the euclidean distance between two points
def count_euclidean_distance(point1 ,point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point2[i] - point1[i]) ** 2
    return np.sqrt(distance)

# Get the euclidean version of analogy z = z + (y - x), a new vector
def get_analogy_vector(xVector, yVector, zVector):
    return zVector + (xVector - yVector)

# Get the closest word to the given vector that is not present in words
def get_closest_words(vector, words):
    closest_words = {}
    for word, idx in vocab.items():
        if word in words:
            continue
        
        word_vector = W[idx]
        distance = count_euclidean_distance(vector, word_vector)
        if len(closest_words) <= 1:
            closest_words[word] = distance
        else:
            biggest_distance_word = max(closest_words, key=closest_words.get)
            biggest_distance = closest_words[biggest_distance_word]
            if biggest_distance > distance:
                del closest_words[biggest_distance_word]
                closest_words[word] = distance
            
    return closest_words

def get_words(words):
    vectorX = W[vocab[words[0]]]
    vectorY = W[vocab[words[1]]]
    vectorZ = W[vocab[words[2]]]
    newVector = get_analogy_vector(vectorX, vectorY, vectorZ)
    closest_words = get_closest_words(newVector, words)
    return closest_words

    
# Main loop for analogy
while True:
    input_terms = input("\nEnter 3 words separated by '-' (EXIT or words != 3): ")
    if input_terms == 'EXIT':
        break
    else:
        words = input_terms.split("-")
        
        # Break if exactly 3 words not present
        if len(words) != 3:
            break
        
        # Find the two words closest to the third word similarly as the first two words      
        closest = get_words(words)
        
        # Sort the dictionary based on the distance which is value in dict
        sorted_closest = {k: v for k, v in sorted(closest.items(), key=lambda item: item[1])}
        print("\n%s is to %s as %s is to \n" % (words[0], words[1] , words[2]))
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for word, distance in sorted_closest.items():
            print("%35s\t\t%f\n" % (word, distance))