from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict



######### OOV WORDS ANALYSIS

def get_OOV_analysis(embedding_model : Dict[str, np.array], texts: List[List[str]]):
    """_summary_

    Parameters
    ----------
    embedding_model : Dict[str, np.array]
        _description_
    texts : List[List[str]]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n_words = len(set([word for text in texts for word in text]))
    OOV_words_list = [word for text in texts for word in text if word not in embedding_model]
    OOV_words = set(OOV_words_list)
    n_OOV_words = len(OOV_words)
    proportion_OOV_words = n_OOV_words/n_words
    OOV_words_counts = Counter(OOV_words_list)
    OOV_words_counts = sorted(OOV_words_counts.items(), key=lambda x:x[1])[::-1]

    return OOV_words, proportion_OOV_words, OOV_words_counts



########## OOV WORDS TRAINING

def _get_co_occurrence_matrix(n_tokens : int, token2int: Dict[str, int], sentences: List[List[str]], window_size: int = 5) -> np.ndarray:
    # Create the tokens co-occurence matrix filled with zeros
    co_occurrence_matrix = np.zeros(shape=(n_tokens, n_tokens), dtype=np.int32)

    for sentence in sentences:
        for i, token in enumerate(sentence[:-1]):
            context_token_indices = [token2int[t] for t in sentence[i+1:i+window_size+1]]
            curr_token_index = token2int[token]
            co_occurrence_matrix[context_token_indices, curr_token_index] += 1
            co_occurrence_matrix[curr_token_index, context_token_indices] += 1

    # Force co-occurrences between the same tokens at 0.
    np.fill_diagonal(co_occurrence_matrix, 0)

    # co-occurence matrix is similar
    assert np.all(co_occurrence_matrix.T == co_occurrence_matrix), 'The Co-occurrence matrix is not similar.'

    return co_occurrence_matrix

def _f(x, alpha=3/4, x_max = 100):
    return torch.clip(torch.pow(x/x_max, alpha), 0, 1)

def _loss_function(weights, weights_t, bias, bias_t, co_occurence_matrix):
    formula = _f(co_occurence_matrix) * torch.square(weights @ weights_t.T + bias + bias_t[:,None] - torch.log(1 + co_occurence_matrix))
    loss = torch.sum(formula)
    return loss


def _train_oov_terms(embedding_model, co_occurrence_matrix, token2int, int2token, n_epochs = 100, device='cpu'):
    n_tokens = len(token2int)
    
    embedding_size = list(embedding_model.values())[0].shape[0]

    # Variables declaration
    knownIndices = np.array([token2int[key] if key in token2int.keys() else -1 for key in embedding_model.keys()], dtype=np.int32)
    knownEmbeddings = np.array([embedding_model[int2token[k]] for k in knownIndices[knownIndices!=-1]], dtype=np.float32)

    weights = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias = torch.randn((n_tokens, ), requires_grad=True, device=device)
    weights_t = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias_t = torch.randn((n_tokens, ), requires_grad=True, device=device)

    co_occurrence_matrix=torch.tensor(co_occurrence_matrix, dtype=torch.int64, device=device)

    mask=torch.ones((n_tokens, embedding_size), dtype=torch.int32, device=device)
    k_embedding=torch.zeros((n_tokens, embedding_size), dtype=torch.float32, device=device)

    knownIndices = torch.tensor(knownIndices, dtype=torch.long,  device=device)
    knownEmbeddings = torch.tensor(knownEmbeddings, dtype=torch.float32,  device=device)

    mask[knownIndices[knownIndices != -1]] = 0
    k_embedding[knownIndices[knownIndices != -1]] = knownEmbeddings.clone().detach()


    # Set optimizer
    optimizer=torch.optim.Adam([weights, weights_t, bias, bias_t], lr=1)

    losses = []

    # Training
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # zero the parameter gradients

        optimizer.zero_grad()

        if knownIndices is not None:
            weights_T=mask*weights_t + (1-mask)* (k_embedding-weights)

        # forward + backward + optimize
        loss = _loss_function(weights, weights_T, bias, bias_t, co_occurrence_matrix)
        loss.backward()
        optimizer.step()

        
        losses.append(loss.detach().cpu().numpy())

        if epoch % 10 == 9:
            print('epochs:', epoch + 1, 'loss:', loss.detach().cpu().numpy())

    if knownIndices is not None:
        weights_T=mask*weights_t + (1-mask)* (k_embedding-weights)

    return (weights+weights_T).detach().cpu().numpy(), losses


def extend_embedding_model(embedding_model : Dict[str, np.array], texts: List[List[str]], window_size : int = 5,  n_epochs : int = 100, 
                           device : str = 'cpu'):
    #texts_list = [sentence.lower() for sentence in texts]
    tokens = set([t for text in texts for t in text])
    n_tokens = len(tokens)
    token2int = dict(zip(tokens, range(len(tokens))))
    int2token = {v: k for k,v in token2int.items()}

    print('Building co-occurence matrix...')
    co_occurrence_matrix = _get_co_occurrence_matrix(n_tokens, token2int, texts, window_size=window_size)
    print('Co-occurence matrix shape:', co_occurrence_matrix.shape)
    print()

    print('Training OOV words...')
    weights, losses = _train_oov_terms(embedding_model, co_occurrence_matrix, token2int, int2token, device=device, 
                                       n_epochs=n_epochs)

    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.show()

    extended_embedding_model = { t: weights[i] for i, t in enumerate(tokens) }
    extended_embedding_model.update(embedding_model)

    return extended_embedding_model