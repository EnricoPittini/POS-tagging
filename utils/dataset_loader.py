from typing import Tuple, List

import string
import os

import re

def _build_dataset_from_files_list(file_name_list : List[str], divide_by_sentence : bool = True,
                                   group_numbers : bool = True):
    """Build the dataset from the given list of documents.

    Parameters
    ----------
    file_name_list : List[str]
        List of the documents path names
    divide_by_sentence : bool, optional
        Whether to divide the texts by sentences or by documents, by default True
    group_numbers : bool, optional
        If True, as the default value, the different possible token numbers are grouped into the same special token [num].
        This makes sense since all the numbers are associated to the same POS tag.

    Returns
    -------
    texts : list of str
        List of strings, representing the sentences/documents, i.e. sequence of words.
    labels : list of str
        List of strings, where each string contains the sequence of POS tags for the words in the corresponding 
        sentence/document.
    """
    texts = []
    labels = []

    if not divide_by_sentence:  # Divide by documents
        for document_index, file_name in enumerate(file_name_list):
            texts.append([])
            labels.append([])
            with open(file_name) as f:
                for line in f:
                    split_line = line.split('\t')
                    if len(split_line) < 3:
                        continue
                    word, label, _ = split_line
                    word = word.lower()
                    texts[document_index].append(word)
                    labels[document_index].append(label)

    else:  # Divide by sentences
        document_index = 0
        texts.append([])
        labels.append([])
        for file_name in file_name_list:
            with open(file_name) as f:
                for line in f:
                    split_line = line.split('\t')
                    if len(split_line) < 3:
                        continue
                    word, label, _ = split_line
                    word = word.lower()
                    texts[document_index].append(word)
                    labels[document_index].append(label)
                    if word in ['.', '!', '?']:  # End of the sentence
                        texts.append([])
                        labels.append([])
                        document_index += 1
        texts = texts[:-1]
        labels = labels[:-1]

    texts = [' '.join(list_of_words) for list_of_words in  texts]
    labels = [' '.join(list_of_labels) for list_of_labels in  labels]

    if group_numbers:
        texts = _substitute_numeric(texts)

    return texts, labels
    

def _substitute_numeric(texts):
    """Substitue each number token into the special token [num].

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    texts : list of str
    """
    # The word is either an integer number or a flaoting point (with either . or ,) or two integers divided by "\/".
    pattern = '^[0-9]+((\\\\\/|\,|\.)[0-9]*)?$'
    substitution = '[num]'
    
    return [' '.join([re.sub(pattern, substitution, word) for word in sentence.split(' ')]) for sentence in texts]


def load_datasets(folder_path: str, split_range: Tuple[int,int] = (100, 150), divide_by_sentence: bool = True,
                  group_numbers : bool = True):
    """Load the train-val-test datasets.

    Parameters
    ----------
    folder_path : str
        Path containing the documents.
    split_range : Tuple[int,int], optional
        Tuple specifying the last training document index and the last validation document index, by default (100, 150)
    divide_by_sentence : bool, optional
        Whether to divide the texts by sentences or by documents, by default True
    group_numbers : bool, optional
        If True, as the default value, the different possible token numbers are grouped into the same special token [num].
        This makes sense since all the numbers are associated to the same POS tag.

    Returns
    -------
    (texts_train, labels_train) : tuple
        `texts_train` is a list of strings, where each string is a sentence/document, i.e. sequence of words.
        `labels_train` is a list of strings, where each string contains the sequence of POS tags for the words in the 
        corresponding sentence/document.
    (texts_val, labels_val) : tuple
        `texts_val` is a list of strings, where each string is a sentence/document, i.e. sequence of words.
        `labels_val` is a list of strings, where each string contains the sequence of POS tags for the words in the 
        corresponding sentence/document.
    (texts_test, labels_test) : tuple
        `texts_test` is a list of strings, where each string is a sentence/document, i.e. sequence of words.
        `labels_test` is a list of strings, where each string contains the sequence of POS tags for the words in the 
        corresponding sentence/document.
    """
    # TODO regex
    file_name_list = sorted([os.path.join(folder_path,file_name) 
                            for file_name in os.listdir('dataset') if 'wsj' in file_name and '.dp' in file_name])

    file_name_list_train = file_name_list[0:split_range[0]]
    file_name_list_val = file_name_list[split_range[0]:split_range[1]]
    file_name_list_test = file_name_list[split_range[1]:]

    texts_train, labels_train = _build_dataset_from_files_list(file_name_list_train, divide_by_sentence=divide_by_sentence,
                                                               group_numbers=group_numbers)
    texts_val, labels_val = _build_dataset_from_files_list(file_name_list_val, divide_by_sentence=divide_by_sentence,
                                                           group_numbers=group_numbers)
    texts_test, labels_test = _build_dataset_from_files_list(file_name_list_test, divide_by_sentence=divide_by_sentence,
                                                             group_numbers=group_numbers)

    return (texts_train, labels_train), (texts_val, labels_val), (texts_test, labels_test)

    