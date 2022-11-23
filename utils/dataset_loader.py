from typing import Tuple

import string
import os

def _build_dataset_from_files_list(file_name_list, divide_by_sentence=True, remove_punctuation=True):
    dataset_texts = []
    dataset_labels = []

    if not divide_by_sentence:  # Divide by documents
        for document_index, file_name in enumerate(file_name_list):
            dataset_texts.append([])
            dataset_labels.append([])
            with open(file_name) as f:
                for line in f:
                    split_line = line.split('\t')
                    if len(split_line) < 3:
                        continue
                    word, label, _ = split_line
                    word = word.lower()
                    if remove_punctuation and all([c in string.punctuation for c in word]):  # TODO improve this check
                        continue
                    dataset_texts[document_index].append(word)
                    dataset_labels[document_index].append(label)

    else:  # Divide by sentences
        document_index = 0
        dataset_texts.append([])
        dataset_labels.append([])
        for file_name in file_name_list:
            with open(file_name) as f:
                for line in f:
                    split_line = line.split('\t')
                    if len(split_line) < 3:
                        continue
                    word, label, _ = split_line
                    word = word.lower()
                    if remove_punctuation and all([c in string.punctuation for c in word]):  # TODO improve this check
                        if divide_by_sentence and word in ['.', '!', '?']:
                            dataset_texts.append([])
                            dataset_labels.append([])
                            document_index += 1
                        continue
                    dataset_texts[document_index].append(word)
                    dataset_labels[document_index].append(label)
        dataset_texts = dataset_texts[:-1]
        dataset_labels = dataset_labels[:-1]

    dataset_texts = [' '.join(list_of_words) for list_of_words in  dataset_texts]
    dataset_labels = [' '.join(list_of_labels) for list_of_labels in  dataset_labels]
    return dataset_texts, dataset_labels

def load_datasets(folder_path: str, split_range: Tuple[int,int] = (100, 150), divide_by_sentence: bool = True,
                  remove_punctuation: bool = True):
    # TODO regex
    file_name_list = sorted([os.path.join(folder_path,file_name) for file_name in os.listdir('dataset') if 'wsj' in file_name and '.dp' in file_name])

    file_name_list_train = file_name_list[0:split_range[0]]
    file_name_list_val = file_name_list[split_range[0]:split_range[1]]
    file_name_list_test = file_name_list[split_range[1]:]

    dataset_documents_train, dataset_labels_train = _build_dataset_from_files_list(file_name_list_train, 
                                                                                   remove_punctuation=remove_punctuation,
                                                                                   divide_by_sentence=divide_by_sentence)
    dataset_documents_val, dataset_labels_val = _build_dataset_from_files_list(file_name_list_val, 
                                                                                   remove_punctuation=remove_punctuation,
                                                                                   divide_by_sentence=divide_by_sentence)
    dataset_documents_test, dataset_labels_test = _build_dataset_from_files_list(file_name_list_test, 
                                                                                   remove_punctuation=remove_punctuation,
                                                                                   divide_by_sentence=divide_by_sentence)

    return ((dataset_documents_train, dataset_labels_train), (dataset_documents_val, dataset_labels_val), 
            (dataset_documents_test, dataset_labels_test))

    