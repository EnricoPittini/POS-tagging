import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from collections import OrderedDict

from utils.vocabulary_builder import create_vocabulary

def _get_vocabulary_dictionaries(vocabulary: np.ndarray):
    int2token = {i:token for i, token in enumerate(vocabulary)}
    token2int = {token:i for i, token in int2token.items()}
    return int2token, token2int


def _get_co_occurrence_matrix(vocabulary: np.ndarray, texts: List[List[str]], window_size: int = 5) -> np.ndarray:
    n_tokens = len(vocabulary)

    _, token2int = _get_vocabulary_dictionaries(vocabulary)

    # Create the tokens co-occurence matrix filled with zeros
    co_occurrence_matrix = np.zeros(shape=(n_tokens, n_tokens), dtype=np.int32)

    for sentence in texts:
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


def _train_oov_terms(embedding_model, co_occurrence_matrix, vocabulary : np.ndarray, previous_vocabulary : np.ndarray = None,
                     previous_trained_parameters_dict : dict = None, n_epochs = 100, device='cpu', adam_slow_start=0.5):


    n_tokens = len(vocabulary)
    int2token, token2int = _get_vocabulary_dictionaries(vocabulary)
    
    embedding_size = list(embedding_model.values())[0].shape[0]

    # Variables declaration
    knownIndices = np.array([token2int[key] if key in token2int.keys() else -1 for key in embedding_model.keys()], dtype=np.int32)
    knownEmbeddings = np.array([embedding_model[int2token[k]] for k in knownIndices[knownIndices!=-1]], dtype=np.float32)

    weights = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias = torch.randn((n_tokens, ), requires_grad=True, device=device)
    weights_t = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias_t = torch.randn((n_tokens, ), requires_grad=True, device=device)


    # Set optimizer
    optimizer=torch.optim.Adam([weights, weights_t, bias, bias_t], lr=1)

    if previous_vocabulary is not None:
        previous_int2token, previous_token2int = _get_vocabulary_dictionaries(previous_vocabulary)
        previous_weights, previous_weights_t, previous_bias, previous_bias_t = (previous_trained_parameters_dict['weights'], 
                                                                                previous_trained_parameters_dict['weights_t'], 
                                                                                previous_trained_parameters_dict['bias'], 
                                                                                previous_trained_parameters_dict['bias_t'])

        indices=np.array([token2int[word] for word in previous_vocabulary])

        with torch.no_grad():
            weights[indices] = previous_weights
            weights_t[indices] = previous_weights_t
            bias[indices] = previous_bias
            bias_t[indices] = previous_bias_t

        dict = previous_trained_parameters_dict['optim'].state_dict()

        for (i, token) in dict['state'].items():
            
            if len(token['exp_avg'].shape)==2:
                exp_average=torch.zeros((n_tokens, embedding_size), device=device)
                exp_avg_sq=torch.zeros((n_tokens, embedding_size), device=device) + adam_slow_start
            else:
                exp_average=torch.zeros((n_tokens,),device=device)
                exp_avg_sq=torch.zeros((n_tokens,),device=device) + adam_slow_start

            exp_average[indices]=token['exp_avg']
            exp_avg_sq[indices]=token['exp_avg_sq']
            
            dict['state'][i]['exp_avg']=exp_average
            dict['state'][i]['exp_avg_sq']=exp_avg_sq
            
        optimizer.load_state_dict(dict)

    co_occurrence_matrix=torch.tensor(co_occurrence_matrix, dtype=torch.int64, device=device)

    mask=torch.ones((n_tokens, embedding_size), dtype=torch.int32, device=device)
    k_embedding=torch.zeros((n_tokens, embedding_size), dtype=torch.float32, device=device)

    knownIndices = torch.tensor(knownIndices, dtype=torch.long,  device=device)
    knownEmbeddings = torch.tensor(knownEmbeddings, dtype=torch.float32,  device=device)

    mask[knownIndices[knownIndices != -1]] = 0
    k_embedding[knownIndices[knownIndices != -1]] = knownEmbeddings.clone().detach()


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

    trained_parameters_dict = {'weights': weights, 'weights_t': weights_T, 'bias': bias, 'bias_t':bias_t, 'optim': optimizer}

    return (weights+weights_T).detach().cpu().numpy(), losses, trained_parameters_dict


def extend_embedding_model( embedding_model : Dict[str, np.array], 
                            texts: List[List[str]], 
                            previous_vocabulary : np.ndarray = None,
                            previous_trained_parameters_dict : dict = None, 
                            window_size : int = 5,  
                            n_epochs : int = 100, 
                            device : str = 'cpu'):

    if (previous_vocabulary is not None and previous_trained_parameters_dict is None) or (previous_vocabulary is None and previous_trained_parameters_dict is not None):
        raise ValueError('`previous_vocabulary` and `previous_trained_parameters_dict` must be either both None or not None')

    #texts_list = [sentence.lower() for sentence in texts]
    vocabulary, _ = create_vocabulary(texts, padding=False)
    """tokens = set([t for text in texts for t in text])
    n_tokens = len(tokens)
    token2int = dict(zip(tokens, range(len(tokens))))
    int2token = {v: k for k,v in token2int.items()}"""
    n_tokens = len(vocabulary)
    int2token = {i:token for i, token in enumerate(vocabulary)}
    token2int = {token:i for i, token in int2token.items()}

    print('Building co-occurence matrix...')
    co_occurrence_matrix = _get_co_occurrence_matrix(vocabulary, texts, window_size=window_size)
    print('Co-occurence matrix shape:', co_occurrence_matrix.shape)
    print()

    print('Training OOV words...')
    weights, losses, trained_parameters_dict = _train_oov_terms(embedding_model, co_occurrence_matrix, vocabulary,
                                                                previous_vocabulary, previous_trained_parameters_dict, device=device, 
                                                                n_epochs=n_epochs)

    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.show()

    extended_embedding_model = { t: weights[i] for i, t in enumerate(vocabulary) }
    extended_embedding_model.update(embedding_model)

    return extended_embedding_model, vocabulary, trained_parameters_dict