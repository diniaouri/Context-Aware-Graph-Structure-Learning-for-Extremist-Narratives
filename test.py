import pickle


def print_matrix_from_pickle(file_path):
    # Load the matrix from the pickle file
    with open(file_path, 'rb') as file:
        matrix = pickle.load(file)

    # Print the matrix
    for row in matrix:
        print(row)


# Example usage
# Replace with your actual file path
pickle_file_path = 'adjacency_matrices/adjacency_learned_epoch_1000_exp6.pkl'
print_matrix_from_pickle(pickle_file_path)
