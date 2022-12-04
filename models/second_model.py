import tensorflow.keras as ks

def build_second_model(n_classes, embedding_dim=50, embedding_matrix=None, latent_dim=128, dropout=0.0, merge_mode='concat'):
    """Build the first model, consisting in two BiLSTM layers plus a Dense layer on top.

    Parameters
    ----------
    n_classes : int
        Number of classes (i.e. number of POS tags)
    embedding_dim : int, optional
        Dimensionality of the embeddings, by default 50
    embedding_matrix : np.array, optional
        Bidimensional matrix containing the words embeddings, by default None
    latent_dim : int, optional
        Dimensionality of the hidden states returned by each LSTM, by default 128.
        Since the two LSTM are concatenated, the actual hidden states returned by the BiLSTM have dimensionality 2*`latent_dim`.    

    Returns
    -------
    ks.Model
        The model.
    """    
    inputs = ks.layers.Input(shape=(None,))

    V = embedding_matrix.shape[0]
    embeddings =   ks.layers.Embedding(output_dim=embedding_dim, input_dim=V, 
                        weights=embedding_matrix if embedding_matrix is None else [embedding_matrix],
                        mask_zero=True,
                        trainable=False)(inputs)

    # Application of the first Bidirectional LSTM. We take the full sequence of hidden states `h_outputs`.
    h_outputs, _, _, _, _ = ks.layers.Bidirectional(ks.layers.LSTM(units=latent_dim, return_sequences=True, 
                                                        return_state=True, dropout=dropout))(embeddings)

    # Application of the second Bidirectional LSTM. We take the full sequence of hidden states `h_outputs`.
    h_outputs, _, _, _, _ = ks.layers.Bidirectional(ks.layers.LSTM(units=latent_dim, return_sequences=True, 
                                                        return_state=True, dropout=dropout), merge_mode=merge_mode)(h_outputs)

    # Final outputs of the model: for each input token, we produce a distribution for predicting its POS tag. 
    # No activation function is used (`SparseCategoricalCrossentropy(from_logits=True)` is used as loss).
    outputs = ks.layers.Dense(units=n_classes)(h_outputs)

    model = ks.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', 
                       metrics=['accuracy'])

    return model



