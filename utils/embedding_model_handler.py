import numpy as np
import os
from typing import Dict
from tqdm import tqdm

def load_embedding_model(folder_path : str, embedding_dim : int = 50, extended_version : bool = False):
    """Load the GLOVE embeddings.

    Parameters
    ----------
    folder_path : str
        Path of the folder containing the GLOVE embeddings.
    embedding_dim : int, optional
        Embedding dimension, by default 50
    extended_version : bool, optional
        Whether to use the extendend GLOVE embeddings, covering also the OOV words of our dataset, or not, by default False.

    Returns
    -------
    GLOVE_embeddings : dict
        Dictionary mapping word types into np.array embedding vectors.
    """
    GLOVE_embeddings = []

    if not extended_version:
        file_path = os.path.join(folder_path, f'glove.6B.{embedding_dim}d.txt')
    else:
        file_path = os.path.join(folder_path, f'extended_glove.{embedding_dim}d.txt')

    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().splitlines()
        GLOVE_embeddings += text

    GLOVE_embeddings = [line.split() for line in GLOVE_embeddings if len(line)>0]
    GLOVE_embeddings = {line[0]:np.array(line[1:],dtype=np.float32) for line in GLOVE_embeddings}

    return GLOVE_embeddings


def store_embedding_model(embedding_path : str, embedding_model : Dict[str, np.array]):
    with open(embedding_path, 'w', encoding='utf-8') as file:
        for token,embedding in tqdm(embedding_model.items()):
            try:
                embedding = ' '.join(str(embedding).split('\n'))
                file.write(f'{token} {embedding}\n'.replace('[', '').replace(']', ''))
            except Exception as e:
                print(f'{token} {embedding}'.replace('[', '').replace(']', ''))
                raise e
