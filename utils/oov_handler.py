import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from utils.vocabulary_builder import create_vocabulary


def _get_vocabulary_dictionaries(vocabulary: np.ndarray):
    """Return the dictionaries representing the mapping from integer id to token and the reverse mapping.

    Parameters
    ----------
    vocabulary : np.ndarray
        Array of strings, representing the mapping from integer id to token.

    Returns
    -------
    int2token : dict
        Mapping from integer id to token.
    token2int : dict
        Mapping from token to integer id.
    """
    int2token = {i:token for i, token in enumerate(vocabulary)}
    token2int = {token:i for i, token in int2token.items()}
    return int2token, token2int


def _get_co_occurrence_matrix(vocabulary: np.ndarray, texts: List[List[str]], window_size: int = 5) -> np.ndarray:
    """Return the co-occurence matrix built from the given texts.

    The co-occurence matrix is a term-term matrix, containing the co-occurance counts between each pair of tokens.

    Parameters
    ----------
    vocabulary : np.ndarray
        Array of strings, representing the mapping from integer id to token.
    texts : List[List[str]]
        Each list is a sentence, represented as a list of words/strings.
    window_size : int, optional
        Size of the window, used for counting the co-occurances of the tokens, by default 5

    Returns
    -------
    co_occurrence_matrix : np.ndarray
        Bidimensional array representing the co-occurence matrix
    """
    n_tokens = len(vocabulary)

    _, token2int = _get_vocabulary_dictionaries(vocabulary)

    # Create the tokens co-occurence matrix filled with zeros
    co_occurrence_matrix = np.zeros(shape=(n_tokens, n_tokens), dtype=np.int32)

    # Fill the co-occurence matrix
    for sentence in texts:
        for i, token in enumerate(sentence[:-1]):
            context_token_indices = [token2int[t] for t in sentence[i+1:i+window_size+1]]
            curr_token_index = token2int[token]
            co_occurrence_matrix[context_token_indices, curr_token_index] += 1
            co_occurrence_matrix[curr_token_index, context_token_indices] += 1

    # Force co-occurrences between the same tokens at 0.
    np.fill_diagonal(co_occurrence_matrix, 0)

    # Check that co-occurence matrix is similar
    assert np.all(co_occurrence_matrix.T == co_occurrence_matrix), 'The Co-occurrence matrix is not similar.'

    return co_occurrence_matrix


def _f(x, alpha=3/4, x_max = 100):
    return torch.clip(torch.pow(x/x_max, alpha), 0, 1)

def _loss_function(weights, weights_t, bias, bias_t, co_occurence_matrix):
    formula = _f(co_occurence_matrix) * torch.square(weights @ weights_t.T + bias + bias_t[:,None] - torch.log(1 + co_occurence_matrix))
    loss = torch.sum(formula)
    return loss


def _train_oov_terms(embedding_model : Dict[str,np.ndarray], co_occurrence_matrix : np.ndarray, vocabulary : np.ndarray, 
                     previous_vocabulary : np.ndarray = None, previous_trained_parameters_dict : dict = None, n_epochs = 100,
                     device='cpu'):
    """Train the OOV words embeddings using the same procedure described in the GloVe paper.

    The OOV words are the words present in the texts, i.e. present in `co_occurrence_matrix`, but not in `embedding_model`.

    The OOV words embeddings are first initialized as random, then are trained, using the GloVe procedure. As described
    in the GloVe paper, this training is based on using the matrices of weigths $W$ and $\tilde{W}$ and the vectors of biases
    $b$ and $\tilde{b}$. In the end of the training, the embedding matrix is the elment-wise sum $W+\tilde{W}$.

    It is very important to point out that only the OOV words embeddings are trained, while the embeddings of all other words
    (i.e. the knows tokens) are kept frozen, meaning that the training does not change them.

    This training is supposed to be performed in the following way. Basically, three different calls of this procedure are 
    performed.
    1. First of all, the OOV words of the training set are trained.
    2. Then, the OOV words of the validation set are trained. The parameters $W$, $\tilde{W}$, $b$ and $\tilde{b} returned by
    the previous call are used to initialize the same parameters of this current call. In other words, the parameters of
    the OOV words are initialized as random, while the known tokens are initialized using the final values of the previous call.
    3. Finally, the OOV words of the test set are trained. As just explained, also here the parameters returned by the 
    previous call are used to initialize the parameters of this current call.    

    Parameters
    ----------
    embedding_model : Dict[str,np.ndarray]
        Mapping from tokens to the corresponding embedding vectors.
    co_occurrence_matrix : np.ndarray
        Co-occurence matrix, built from the texts.
        It is used for the GloVe training procedure.
    vocabulary : np.ndarray
        Array of strings, representing the mapping from integer id to token.
    previous_vocabulary : np.ndarray, optional
        Vocabulary used in the previous call, represented as a mapping from integr id to token.
        By default None, meaning that no previous call has been performed (i.e. we are training the OOV words of the training
        set).
    previous_trained_parameters_dict : dict, optional
        Dictionary containing the parameters returned by the previous call. namely, the keys of this dictionary are: 'weigths',
        'weigths_t', 'bias', 'bias_t'.
        By default None, meaning that no previous call has been performed (i.e. we are training the OOV words of the training
        set).
    n_epochs : int, optional
        Number of training epochs, by default 100
    device : str, optional
        Device on which attach the training, by default 'cpu'

    Returns
    -------
    embedding_matrix : np.ndarray
        Bidimensional array, whose rows represent the token ids, containing the words embeddings.
        The embeddings of the words which were already present in `embedding_model` are unchanged, while the embeddings of the
        OOV words are the only trained.
    losses : list
        List of epoch losses
    trained_parameters_dict : dict
        Dictionary containing the parameters tensors. Namely the keys are: 'weigths', 'weigths_t', 'bias', 'bias_t'.
    """
    # Number of tokens in our texts
    n_tokens = len(vocabulary)
    int2token, token2int = _get_vocabulary_dictionaries(vocabulary)
    
    # Number of tokes in the embedding model
    embedding_size = list(embedding_model.values())[0].shape[0]

    # Array indicating, for each token in our texts, if it is known or OOV. The known tokens are represented with their 
    # integer id, while the OOV are represented as -1
    knownIndices = np.array([token2int[key] if key in token2int.keys() else -1 for key in embedding_model.keys()], dtype=np.int32)
    # Array containing the embeddings of the known tokens
    knownEmbeddings = np.array([embedding_model[int2token[k]] for k in knownIndices[knownIndices!=-1]], dtype=np.float32)
    knownIndices = torch.tensor(knownIndices, dtype=torch.long,  device=device)
    knownEmbeddings = torch.tensor(knownEmbeddings, dtype=torch.float32,  device=device)
    k_embedding=torch.zeros((n_tokens, embedding_size), dtype=torch.float32, device=device)      
    k_embedding[knownIndices[knownIndices != -1]] = knownEmbeddings.clone().detach()

    # Initialization of the parameters. Random initialization.
    weights = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias = torch.randn((n_tokens, ), requires_grad=True, device=device)
    weights_t = torch.randn((n_tokens, embedding_size), requires_grad=True, device=device)
    bias_t = torch.randn((n_tokens, ), requires_grad=True, device=device)

    # Set optimizer
    optimizer=torch.optim.Adam([weights, weights_t, bias, bias_t], lr=1e-1)

    if previous_vocabulary is not None:
        # If `previous_vocabulary` and `previous_trained_parameters_dict` are specified, we have to inject into the 
        # parameters of the known tokens the parameters returned by the previous call

        # Parameters returned by the previous call
        previous_weights, previous_weights_t, previous_bias, previous_bias_t = (previous_trained_parameters_dict['weights'], 
                                                                                previous_trained_parameters_dict['weights_t'], 
                                                                                previous_trained_parameters_dict['bias'], 
                                                                                previous_trained_parameters_dict['bias_t'])

        # Indices of the known tokens in the previous vocabulary
        indices=np.array([token2int[word] for word in previous_vocabulary])

        # Injection of the previous parameters into the current ones for the known tokens
        with torch.no_grad():
            weights[indices] = previous_weights
            weights_t[indices] = previous_weights_t
            bias[indices] = previous_bias
            bias_t[indices] = previous_bias_t

        # Injecting the state dict of the optimizer
        adam_slow_start = 0.5
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

    # Boolean mask indicating where are the OOV tokens (1: OOV token; 0: known token)
    mask=torch.ones((n_tokens, embedding_size), dtype=torch.int32, device=device)
    mask[knownIndices[knownIndices != -1]] = 0

    losses = []

    # Training
    for epoch in range(n_epochs):  
        # zero the parameter gradients
        optimizer.zero_grad()

        if knownIndices is not None:
            # IMPORTANT POINT: forcing the known terms embeddings to stay the same 
            weights_T = mask*weights_t + (1-mask)* (k_embedding-weights)

        # forward + backward + optimize
        loss = _loss_function(weights, weights_T, bias, bias_t, co_occurrence_matrix)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.detach().cpu().numpy())

        if epoch % 10 == 9:
            print('epochs:', epoch + 1, 'loss:', loss.detach().cpu().numpy())

    if knownIndices is not None:
        # IMPORTANT POINT: forcing the known terms embeddings to stay the same 
        weights_T = mask*weights_t + (1-mask)* (k_embedding-weights)

    trained_parameters_dict = {'weights': weights, 'weights_t': weights_T, 'bias': bias, 'bias_t':bias_t, 'optim': optimizer}

    # The final embedding matrix is the sum of `weigths` and `weights_T``
    embedding_matrix = (weights+weights_T).detach().cpu().numpy()

    return embedding_matrix, losses, trained_parameters_dict


def extend_embedding_model( embedding_model : Dict[str, np.array], 
                            texts: List[List[str]], 
                            previous_vocabulary : np.ndarray = None,
                            previous_trained_parameters_dict : dict = None, 
                            window_size : int = 5,  
                            n_epochs : int = 100, 
                            device : str = 'cpu'):
    """Extend the given embedding model with the embeddings for the OOV words inside the given texts.

    The OOV words are the words contained in the given texts but not in the embedding model. See the function 
    `_train_oov_terms`.
    It is very important to point out that only the OOV words embeddings are trained, while the embeddings of all other words
    (i.e. the knows tokens) are kept frozen, meaning that the training does not change them.

    The expansion of the embedding model should be performed in the following way, performing three different calls of this
    procedure.
    1. First of all, the standard GloVe embedding model is extended with the training set OOV words.
    2. Then, the current embedding model is further expanded with the validation set OOV words.
    3. Finally, the current embedding model is further expanded with the test set OOV words.

    Parameters
    ----------
    embedding_model : Dict[str,np.ndarray]
        Mapping from tokens to the corresponding embedding vectors.
    texts : List[List[str]]
        Each list is a sentence, represented as a list of words/strings.
    previous_vocabulary : np.ndarray, optional
        Vocabulary used in the previous call, represented as a mapping from integr id to token.
        By default None, meaning that no previous call has been performed (i.e. we are training the OOV words of the training
        set).
    previous_trained_parameters_dict : dict, optional
        Dictionary containing the parameters returned by the previous call. namely, the keys of this dictionary are: 'weigths',
        'weigths_t', 'bias', 'bias_t'.
        By default None, meaning that no previous call has been performed (i.e. we are training the OOV words of the training
        set).
    window_size : int, optional
        Size of the window, used for counting the co-occurances of the tokens, by default 5
    n_epochs : int, optional
        Number of training epochs, by default 100
    device : str, optional
        Device on which attach the training, by default 'cpu'

    Returns
    -------
    extended_embedding_model : Dict[str,np.array]
        Extended embedding model, implemented as a mapping from the tokens to the corresponding embedding vectors
    vocabulary : np.array
        Vocabulary used for the given `texts`, implemented as an array of strings, representing the mapping from the integer
        ids to the corresponding tokens.
    trained_parameters_dict
        Dictionary containing the parameters returned by this OOV training. It contains four keys: `weigths`, `weights_t`, 
        `bias`, `bias_t`. 

    Raises
    ------
    ValueError
        `previous_vocabulary` and `previous_trained_parameters_dict` must be either both None or not None 
    """
    if (previous_vocabulary is not None and previous_trained_parameters_dict is None) or \
         (previous_vocabulary is None and previous_trained_parameters_dict is not None):
        raise ValueError('`previous_vocabulary` and `previous_trained_parameters_dict` must be either both None or not None')

    vocabulary, _ = create_vocabulary(texts, add_padding_token=False)
    int2token = {i:token for i, token in enumerate(vocabulary)}

    print('Building co-occurence matrix...')
    co_occurrence_matrix = _get_co_occurrence_matrix(vocabulary, texts, window_size=window_size)
    print('Co-occurence matrix shape:', co_occurrence_matrix.shape)
    print()

    print('Training OOV words...')
    weights, losses, trained_parameters_dict = _train_oov_terms(embedding_model, co_occurrence_matrix, vocabulary,
                                                                previous_vocabulary, previous_trained_parameters_dict, 
                                                                device=device, 
                                                                n_epochs=n_epochs)

    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.show()

    extended_embedding_model = { t: weights[i] for i, t in enumerate(vocabulary) }
    extended_embedding_model.update(embedding_model)

    return extended_embedding_model, vocabulary, trained_parameters_dict