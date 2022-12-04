import matplotlib.pyplot as plt

def plot_history(history):
    """Plot the training history

    Parameters
    ----------
    history : dict
        History object returned by the `fit` method.
    """
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
    plt.show()