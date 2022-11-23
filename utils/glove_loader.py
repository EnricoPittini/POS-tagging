import numpy as np
import os

def load_glove(folder_path : str, embedding_dim : int = 50):
    """Load the GLOVE embeddings.

    Parameters
    ----------
    folder_path : str
        Path of the folder containing the GLOVE embeddings.
    embedding_dim : int, optional
        Embedding dimension, by default 50

    Returns
    -------
    GLOVE_embeddings : dict
        Dictionary mapping word types into np.array embedding vectors.
    """
    GLOVE_embeddings = []
    file_path = os.path.join(folder_path, f'glove.6B.{embedding_dim}d.txt')
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().splitlines()
        GLOVE_embeddings += text

    GLOVE_embeddings = [line.split() for line in GLOVE_embeddings if len(line)>0]
    GLOVE_embeddings = {line[0]:np.array(line[1:]) for line in GLOVE_embeddings}

    return GLOVE_embeddings
