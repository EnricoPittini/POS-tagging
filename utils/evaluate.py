from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import Counter 

def evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels, show_classification_report : bool = False):
    """Compute the f1 macro, without considering the punctuation

    Parameters
    ----------
    y_true : np.array
    y_pred : np.array
    punctuation_integers : list
        List of the integers corresponding to the punctuation classes (i.e. punctuation POS tags)
    """
    """padding_mask = y_true==0
    y_true = y_true[~padding_mask]
    y_pred = y_pred[~padding_mask]"""
    ids_no_evaluate = [i for i,pos_tag in enumerate(vocabulary_labels) if pos_tag in tags_no_evaluate]

    def f(v):
        return v in ids_no_evaluate
    mask = np.vectorize(f)(y_true)
    mask = np.logical_or(mask, np.vectorize(f)(y_pred))
    #print(mask.shape)
 
    y_true = y_true[~mask]
    y_pred = y_pred[~mask]

    tags_to_evaluate = [tag for i, tag in enumerate(vocabulary_labels) if np.any(y_true==i) or np.any(y_pred==i)]
    class_report = classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=True)

    if show_classification_report:
        print(classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=False))

    return f1_score(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), average='macro'), class_report


def wrongly_classified_tokens_analysis(x, y_true, y_pred, tags_no_evaluate, vocabulary_labels, vocabulary):
    ids_no_evaluate = [i for i,pos_tag in enumerate(vocabulary_labels) if pos_tag in tags_no_evaluate]

    def f(v):
        return v in ids_no_evaluate
    mask = np.vectorize(f)(y_true)
    mask = np.logical_or(mask, np.vectorize(f)(y_pred))
    #print(mask.shape)
 
    y_true = y_true[~mask]
    y_pred = y_pred[~mask]
    x = x[~mask]

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    wrong_class_mask = y_true!=y_pred 
    # y_true = y_true[wrong_class_mask]
    # y_pred = y_pred[wrong_class_mask]
    x = x[wrong_class_mask]

    wrong_tokens_ids = np.array(list(Counter(x).keys())) # equals to list(set(words))
    wrong_tokens_counts = list(Counter(x).values()) # counts the elements' frequency
    wrong_tokens_ids = wrong_tokens_ids[np.argsort(wrong_tokens_counts)][::-1]
    wrong_tokens_counts = sorted(wrong_tokens_counts)[::-1]
    wrong_tokens = vocabulary[wrong_tokens_ids]

    return wrong_tokens, wrong_tokens_counts


def plot_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend()
    plt.title('Loss history')

    plt.figure()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend()
    plt.title('Accuracy history')