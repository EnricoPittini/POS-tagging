import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, texts, labels, batch_size=32, shuffle=True, max_seq_length=None):
        'Initialization'
        if len(texts)!=len(labels):
            raise ValueError('`texts` and `labels` must have the same length')
        self.texts = texts 
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        texts_batch = [self.texts[i] for i in indexes]
        labels_batch = [self.labels[i] for i in indexes]

        # Generate data
        x_batch, y_batch = self.__batch_generation(texts_batch, labels_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.texts))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __batch_generation(self, texts_batch, labels_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        max_seq_length = max([len(text) for text in texts_batch]) if self.max_seq_length is None else self.max_seq_length

        # Initialization
        x_batch = np.empty((self.batch_size, max_seq_length), dtype=int)
        y_batch = np.empty((self.batch_size, max_seq_length), dtype=int)

        #print(texts_batch[0])

        # Generate data
        for i in range(self.batch_size):
            padding = np.zeros((max_seq_length,))[len(texts_batch[i]):]

            # Store sample
            x_batch[i,] = texts_batch[i] + list(padding)

            # Store class
            y_batch[i] = labels_batch[i] + list(padding)

        return x_batch, y_batch