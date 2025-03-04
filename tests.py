import numpy as np
import pickle

embeddings = np.load(
    "./embeddings/All_french_tweet_data_embeddings_exp4.npy")
print(embeddings.shape)
adjacency_file = './adjacency_matrices/adjacency_learned_epoch_1000_exp4.pkl'

with open(adjacency_file, 'rb') as f:
    adj_matrix = pickle.load(f)

print(adj_matrix.shape)
