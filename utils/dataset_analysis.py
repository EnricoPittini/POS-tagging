from collections import Counter, OrderedDict
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

def OOV_analysis(embedding_model : Dict[str, np.array], texts: List[List[str]], plot : bool = False, k : int = 20):
    """Analyse the OOV words, i.e. the words present in `texts` but not in `embedding_model`.

    Parameters
    ----------
    embedding_model : Dict[str, np.array]
        Mapping from words to embedding vectors, e.g. Glove embedding model.
    texts : List[List[str]]
        Each list is a sentence, represented as a list of words/strings.
    plot : bool, optional
        Whether to plot or not the most frequent OOV words, by default False
    k : int, optional
        Number of OOV words to plot, by default 20

    Returns
    -------
    OOV_words : set
        OOV words
    proportion_OOV_words : float
        Proportion of OOV words w.r.t. all words
    OOV_words_counts_dict : OrderedDict
        Mapping from OOV words to their number of instances, sorted in descending order w.r.t. the number of inctances.
    """
    n_words = len(set([word for text in texts for word in text]))
    OOV_words_list = [word for text in texts for word in text if word not in embedding_model]
    OOV_words = set(OOV_words_list)
    n_OOV_words = len(OOV_words)
    proportion_OOV_words = n_OOV_words/n_words
    OOV_words_counts_dict = Counter(OOV_words_list)
    OOV_words_counts_dict = OrderedDict(sorted(OOV_words_counts_dict.items(), key=lambda x:x[1])[::-1])

    if plot:
        plt.figure(figsize=(15,10))
        plt.bar(list(OOV_words_counts_dict.keys())[:k], list(OOV_words_counts_dict.values())[:k])
        plt.xticks(rotation=45)
        plt.ylabel('Number of instances')
        plt.xlabel('OOV words')
        plt.title(f'{k} most frequent OOV words')
        plt.grid(axis='y')
        plt.show()

    return OOV_words, proportion_OOV_words, OOV_words_counts_dict



def most_frequent_tokens_analysis(texts: List[List[str]], plot : bool = False, k : int = 20, xlabel : str = 'Tokens', 
                                  title : str = None):
    """Analyse the most frequent tokens in the given dataset

    Parameters
    ----------
    texts : List[List[str]]
        Each list is a sentence, represented as a list of words/strings.
    plot : bool, optional
        Whether to plot or not the most frequent tokens, by default False
    k : int, optional
        Number of tokens to plot, by default 20
    xlabel : str, optional
        Label on the x axis, by default 'Tokens'
    title : str, optional
        Plot title, by default None

    Returns
    -------
    tokens_counts_dict : OrderedDict
        Mapping tokens -> counts, sorted by frequency.
    """
    tokens_counts_dict = Counter([word for text in texts for word in text])
    tokens_counts_dict = OrderedDict(sorted(tokens_counts_dict.items(), key=lambda x:x[1])[::-1])

    if plot:
        plt.figure(figsize=(15,10))
        plt.bar(list(tokens_counts_dict.keys())[:k], list(tokens_counts_dict.values())[:k])
        plt.xticks(rotation=45)
        plt.ylabel('Number of instances')
        plt.xlabel(xlabel)
        title = f'{k} most frequent tokens' if title is None else title
        plt.title(title)
        plt.grid(axis='y')
        plt.show()

    return tokens_counts_dict


def plot_sequence_length_analysis(texts : List[List[str]]):
    """Analyse the length of the sequences in the given dataset

    Parameters
    ----------
    texts : List[List[str]]
        Each list is a sentence, represented as a list of words/strings.
    """
    # Length of each training sentence
    train_sentences_lenghts = [len(sentence) for sentence in texts]

    # Histogram of the sentences length distribution
    hist, bin_edges = np.histogram(train_sentences_lenghts, bins=np.max(train_sentences_lenghts) + 1, density=True) 
    # Cumulative distribution of the sentences length
    C = np.cumsum(hist)*(bin_edges[1] - bin_edges[0])

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(bin_edges[1:], hist)
    plt.title('Distribution of the sentence length across the train dataset')
    plt.xlabel('Sentence length')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], C)
    plt.title('Comulative distribution of the sentence length across the train dataset')
    plt.xlabel('Sentence length')
    plt.grid()
    plt.show()