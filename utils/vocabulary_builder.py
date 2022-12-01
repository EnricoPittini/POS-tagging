from collections import Counter
from typing import List
import numpy as np

def create_vocabulary(texts : List[List[str]]):
    texts_flat = [word for text in texts for word in text]
    tokens = np.array(list(Counter(texts_flat).keys())) # equals to list(set(words))
    tokens_counts = list(Counter(texts_flat).values()) # counts the elements' frequency
    tokens = tokens[np.argsort(tokens_counts)][::-1]

    vocabulary = np.array([''] +  list(tokens)) # dict({word:id+1 for id, word in enumerate(tokens)})
    # id2word = dict({id:word for word, id in word2id .items()})
    word2id = {word:id for id, word in enumerate(vocabulary)}

    texts_ids = [[word2id[word] for word in text] for text in texts]

    return vocabulary, texts_ids