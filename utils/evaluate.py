from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

def f1_macro(y_true, y_pred, punctuation_integers):
    """Compute the f1 macro, without considering the punctuation

    Parameters
    ----------
    y_true : np.array
    y_pred : np.array
    punctuation_integers : list
        List of the integers corresponding to the punctuation classes (i.e. punctuation POS tags)
    """
    def f(v):
        return v in punctuation_integers
    punctuation_mask_test = np.vectorize(f)(y_true)
 
    y_true_noPunctuation = y_true[~punctuation_mask_test]
    y_pred_noPunctuation = y_pred[~punctuation_mask_test]

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