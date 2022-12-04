import numpy as np
import keras
from typing import List

class DataGenerator(keras.utils.Sequence):
    """Generator for the POS tagging task.

    A batch has structure `(x_batch, y_batch)`. Both `x_batch` and `y_batch` are arrays with shape `batch_size*n_tokens`: the
    first contains the tokens, while the second the POS tags (i.e. the labels). 
    `n_tokens` is the max length of a sentence in the batch. Basically, the padding is performed batch-wise: each batch 
    has potentially a different `n_tokens`.

    Parameters
    ----------
    tokens : List[List[int]]
        Each list is a sentence, represented as the list of contained tokens.
    labels : List[List[str]]
        Each list is a sentence, represented as the list of contained POS tags.
    batch_size : int, optional
        Size of each batch, by default 32
    shuffle : bool, optional
        Whether to shuffle or not the dataset before each epoch, by default True
    max_seq_length : int, optional
        Max length of a sequence, by default None.
        If it is not specified, each batch is padded using the max sequence length in that specific batch.
        Otherwise, each batch is padded with the same length `max_seq_length`.
    """
    def __init__(self, tokens : List[List[str]], labels : List[List[str]], batch_size : int = 32, shuffle : bool = True, 
                 max_seq_length : int = None):
        if len(tokens)!=len(labels):
            raise ValueError('`tokens` and `labels` must have the same length')
        self.tokens = tokens 
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch"""
        return int(np.floor(len(self.tokens) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate sentences (both tokens and labels) in the batch
        tokens_batch = [self.tokens[i] for i in indexes]
        labels_batch = [self.labels[i] for i in indexes]

        # Generate batch
        x_batch, y_batch = self.__batch_generation(tokens_batch, labels_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        self.indexes = np.arange(len(self.tokens))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __batch_generation(self, tokens_batch, labels_batch):
        """Generate the batch""" 
        # Sequence length which is used to perform the padding
        max_seq_length = max([len(text) for text in tokens_batch]) if self.max_seq_length is None else self.max_seq_length

        # Initialization
        x_batch = np.empty((self.batch_size, max_seq_length), dtype=int)
        y_batch = np.empty((self.batch_size, max_seq_length), dtype=int)

        # Generate batch
        for i in range(self.batch_size):
            padding = np.zeros((max_seq_length,))[len(tokens_batch[i]):]

            # Store sentence tokens
            x_batch[i,] = tokens_batch[i] + list(padding)

            # Store sentence labels
            y_batch[i] = labels_batch[i] + list(padding)

        return x_batch, y_batch