import numpy as np

from typing import List, Dict

def build_embedding_matrix(vocabulary : List[str], GLOVE_embeddings : Dict[str][np.array], embedding_dimension : int = 50):
    """Build the embedding matrix from the given vocabulary and the GLOVE embeddings,

    Parameters
    ----------
    vocabulary : List[str]
        List of strings, representing the vocabulary: mapping integers -> word types.
    GLOVE_embeddings : Dict[str][np.array]
        Dictionary mapping word types into embedding vectors.
    embedding_dimension : int, optional
        Dimension of the embedding, by default 50

    Returns
    -------
    embedding_matrix : np.array
        The rows are the word integers (given by `vocabulary`), and each row represents the embedding vector of that 
        corresponding word.
    """
    embedding_matrix = np.zeros((len(vocabulary), embedding_dimension), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        try:
            embedding_vector = GLOVE_embeddings[word]
        except (KeyError, TypeError):
            embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

        embedding_matrix[idx] = embedding_vector  

    return embedding_matrix