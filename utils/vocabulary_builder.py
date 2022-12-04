from collections import Counter
from typing import List
import numpy as np

def create_vocabulary(texts : List[List[str]]):
    """Create the vocabulary from  the given texts

    Parameters
    ----------
    texts : List[List[str]]
        Each list is a sentence, represented as a list of strings (i.e. words).

    Returns
    -------
    vocabulary : np.array
        Array of strings, representing the mapping from integer ids to words. 
        The first entry, i.e. index 0, is reserved for the padding: mapping 0 -> ''.
        The entries in this vocabulary are sorted by frequence in descending order: the word with index 1 is the most 
        frequent word.
    texts_ids : List[List[int]]
        It is equal to `texts`, but each word is replaced with the corresponding integer id.

    """
    texts_flat = [word for text in texts for word in text]
    tokens = np.array(list(Counter(texts_flat).keys())) 
    tokens_counts = list(Counter(texts_flat).values()) 
    tokens = tokens[np.argsort(tokens_counts)][::-1]

    vocabulary = np.array([''] +  list(tokens)) 
    word2id = {word:id for id, word in enumerate(vocabulary)}

    texts_ids = [[word2id[word] for word in text] for text in texts]

    return vocabulary, texts_ids