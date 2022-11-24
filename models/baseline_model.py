import tensorflow.keras as ks

def build_baseline_model(sequence_length, n_classes, embedding_dim=50, latent_dim=128, embedding_matrix=None):

    # TODO remove sequence_length
    inputs = ks.layers.Input(shape=(sequence_length,))

    V = embedding_matrix.shape[0]
    embeddings =   ks.layers.Embedding(output_dim=embedding_dim, input_dim=V, input_length=sequence_length, 
                        weights=embedding_matrix if embedding_matrix is None else [embedding_matrix],
                        mask_zero=True)(inputs)

    h_outputs, h_last, c_last = ks.layers.LSTM(units=latent_dim, return_sequences=True, return_state=True)(embeddings)

    outputs = ks.layers.Dense(units=n_classes)(h_outputs)

    model = ks.Model(inputs=inputs, outputs=outputs)

    return model



