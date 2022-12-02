from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import Counter 
import pandas as pd
from collections import OrderedDict


def _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels):
    ids_no_evaluate = [i for i,pos_tag in enumerate(vocabulary_labels) if pos_tag in tags_no_evaluate]

    def f(v):
        return v in ids_no_evaluate
    mask = np.vectorize(f)(y_true)
    mask = np.logical_or(mask, np.vectorize(f)(y_pred))
    #print(mask.shape)
 
    y_true = y_true[~mask]
    y_pred = y_pred[~mask]

    return y_true, y_pred

def compute_f1_score(y_true, y_pred, tags_no_evaluate, vocabulary_labels, show_classification_report : bool = False,
                    k : int = None):
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
    y_true, y_pred = _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)

    return f1_score(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), average='macro')


def compute_class_report(y_true, y_pred, tags_no_evaluate, vocabulary_labels, show : bool = False, k = None, plot : bool = True):
    y_true, y_pred = _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)

    tags_to_evaluate = [tag for i, tag in enumerate(vocabulary_labels) if np.any(y_true==i) or np.any(y_pred==i)]
    class_report = classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=True)

    if show:
        print(classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=False))

    k = len(vocabulary_labels)//2 if k is None else k

    worst_predicted_tags = sorted([(class_report[label]['f1-score'], label) 
                                    for label in class_report.keys() if label not in ['accuracy','macro avg','weighted avg'] ])
    worst_predicted_tags = worst_predicted_tags[:k]
    worst_predicted_tags = OrderedDict({tag:class_report[tag] for (f1_score, tag) in worst_predicted_tags})

    plt.figure(figsize=(12,12))
    x_axis = 2*np.arange(len(worst_predicted_tags))
    plt.bar(x_axis-0.5, [worst_predicted_tags[tag]['precision'] for tag in worst_predicted_tags], label='precision', width=0.5)
    b = plt.bar(x_axis, [worst_predicted_tags[tag]['recall'] for tag in worst_predicted_tags], label='recall', width=0.5)
    plt.bar(x_axis+0.5, [worst_predicted_tags[tag]['f1-score'] for tag in worst_predicted_tags], label='f1-score', width=0.5)
    plt.bar_label(b, labels=[worst_predicted_tags[tag]['support'] for tag in worst_predicted_tags], label_type='center');
    plt.xticks(x_axis, list(worst_predicted_tags.keys()))
    plt.grid(axis='y')
    plt.legend()
    plt.xticks(rotation=45);


    return class_report, worst_predicted_tags


def wrongly_classified_tokens_analysis(x, y_true, y_pred, tags_no_evaluate, vocabulary_labels, vocabulary, k : int = 20,
                                       plot : bool = True):
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
    y_true = y_true[wrong_class_mask]
    y_pred = y_pred[wrong_class_mask]
    x = x[wrong_class_mask]

    wrong_tokens_ids = np.array(list(Counter(x).keys())) # equals to list(set(words))
    wrong_tokens_counts = list(Counter(x).values()) # counts the elements' frequency
    wrong_tokens_ids = wrong_tokens_ids[np.argsort(wrong_tokens_counts)][::-1]
    wrong_tokens_counts = sorted(wrong_tokens_counts)[::-1]
    wrong_tokens = vocabulary[wrong_tokens_ids]

    wrong_tokens_ids, wrong_tokens, wrong_tokens_counts = wrong_tokens_ids[:k], wrong_tokens[:k], wrong_tokens_counts[:k]

    df_entries_list = []
    for wrong_token_id, wrong_token in zip(wrong_tokens_ids,wrong_tokens):
        token_mask = x==wrong_token_id 
        token_tags = vocabulary_labels[np.ravel(y_true[token_mask])]
        pos_tags_counts_dict = dict(Counter(token_tags))
        new_df_entry = pos_tags_counts_dict
        new_df_entry['token'] = wrong_token
        for tag in vocabulary_labels:
            if tag not in new_df_entry:
                new_df_entry[tag] = 0
        df_entries_list.append(new_df_entry)
    wrong_tokens_tags_df = pd.DataFrame(df_entries_list)
    wrong_tokens_tags_df = wrong_tokens_tags_df.set_index(keys='token').fillna(0.0).astype(int)

    plt.figure(figsize=(11,11))
    bottom = [0 for _ in wrong_tokens_tags_df.index]
    for tag in wrong_tokens_tags_df.columns:
        b = plt.bar(x=wrong_tokens_tags_df.index, height=wrong_tokens_tags_df[tag], bottom=bottom)
        bottom += wrong_tokens_tags_df[tag]
        plt.bar_label(b, labels=[tag if wrong_tokens_tags_df[tag][i]>0 else '' for i in range(wrong_tokens_tags_df.shape[0])], label_type='center');
    plt.xticks(rotation=45);
    ax = plt.gca()
    ax.set_ylim([0, sum(wrong_tokens_tags_df.iloc[0])+1]);
    #plt.grid(axis='y');
    plt.ylabel('Number of misclassified instances')
    plt.xlabel('Misclassified tokens')
    plt.title(f'Top {k} misclassified tokens')

    return wrong_tokens, wrong_tokens_counts, wrong_tokens_tags_df


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