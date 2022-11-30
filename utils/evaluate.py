from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def f1_macro(y_true, y_pred, punctuation_integers, show_classification_report : bool = False, tags_to_evaluate : List[str] = None):
    """Compute the f1 macro, without considering the punctuation

    Parameters
    ----------
    y_true : np.array
    y_pred : np.array
    punctuation_integers : list
        List of the integers corresponding to the punctuation classes (i.e. punctuation POS tags)
    """
    padding_mask = y_true==0
    y_true = y_true[~padding_mask]
    y_pred = y_pred[~padding_mask]

    def f(v):
        return v in punctuation_integers
    punctuation_mask_test = np.vectorize(f)(y_true)
 
    y_true_noPunctuation = y_true[~punctuation_mask_test]
    y_pred_noPunctuation = y_pred[~punctuation_mask_test]

    if show_classification_report:
        print(classification_report(y_true=np.ravel(y_true_noPunctuation), y_pred=np.ravel(y_pred_noPunctuation), 
                                    target_names=tags_to_evaluate))

    return f1_score(y_true=np.ravel(y_true_noPunctuation), y_pred=np.ravel(y_pred_noPunctuation), average='macro')


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