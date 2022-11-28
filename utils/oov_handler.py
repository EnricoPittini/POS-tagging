from collections import Counter
import torch
import numpy as np
from typing import List, Dict

def _get_co_occurrences(token: str, token2int: Dict[str, int], sentences: List[List[str]], window_size: int) -> Dict[int, int]:
    # Co-occurrences list for `token`.
    co_occurrences = []
    
    for s in sentences:
        for idx in [i for i, t in enumerate(s) if t == token]:
            co_occurrences.extend(s[max(0, idx - window_size) : min(idx + window_size+1, len(s))])
    
    # Get indices of the tokens in the co_occurrences
    co_occurrence_idxs = [token2int[c] for c in co_occurrences]
    
    # Get a dictionary of number of occurrences per token and sort it by token index.
    co_occurence_dict = Counter(co_occurrence_idxs)
    return dict(sorted(co_occurence_dict.items()))

def get_co_occurrence_matrix(tokens: List[str], token2int: Dict[str, int], sentences: List[List[str]], window_size: int) -> np.ndarray:
    n_tokens = len(tokens)

    # Create the tokens co-occurence matrix filled with zeros
    co_occurrence_matrix = np.zeros(shape=(n_tokens, n_tokens), dtype=np.int32)

    for token in tokens:
        co_occurence_dict = _get_co_occurrences(token, token2int, sentences, window_size)
        token_idx = token2int[token]
        # Update the co-occurrences of other tokens with the given token
        co_occurrence_matrix[token_idx, list(co_occurence_dict.keys())] = list(co_occurence_dict.values())

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

# number of training epochs
#n_epochs = 4000

# tolerance
#eps = 0.001

# number of sentences to consider
#n_sents = 10

# weight embedding size
#embedding_size = 50

# learning rate
#alpha = 0.1

# AdaGrad parameter 
# delta = 0.8

# top N similar words
# topN = 5

def train_oov_terms(GLOVE_embeddings, co_occurrence_matrix, 
                    token2int, int2token, embedding_size = 50, n_epochs = 100):
    n_tokens = len(token2int)
    
    n_tokens = len(token2int)

    # Variables declaration
    knownIndices = np.array([token2int[key] if key in token2int.keys() else -1 for key in GLOVE_embeddings.keys()], dtype=np.int32)
    knownEmbeddings = np.array([GLOVE_embeddings[int2token[k]] for k in knownIndices[knownIndices!=-1]], dtype=np.float32)

    weights = torch.randn((n_tokens, embedding_size), requires_grad=True, device='cuda')
    bias = torch.randn((n_tokens, ), requires_grad=True, device='cuda')
    weights_t = torch.randn((n_tokens, embedding_size), requires_grad=True, device='cuda')
    bias_t = torch.randn((n_tokens, ), requires_grad=True, device='cuda')

    co_occurrence_matrix=torch.tensor(co_occurrence_matrix, dtype=torch.int64, device='cuda')

    mask=torch.ones((n_tokens, embedding_size), dtype=torch.int32, device='cuda')
    k_embedding=torch.zeros((n_tokens, embedding_size), dtype=torch.float32, device='cuda')

    knownIndices = torch.tensor(knownIndices, dtype=torch.long,  device='cuda')
    knownEmbeddings = torch.tensor(knownEmbeddings, dtype=torch.float32,  device='cuda')

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