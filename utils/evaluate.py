from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import Counter 
import pandas as pd
from collections import OrderedDict
from colorama import Fore, Style


def _compute_mask_tags_no_evaluate(y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], 
                                   vocabulary_labels : np.array) -> np.array:
    """Compute the boolean mask localizing the tokens to not evaluate.

    Parameters
    ----------
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.

    Returns
    -------
    mask : np.array
        Boolean mask, with same shape of `y_true` and `y_pred`, localizing the tokens to not evaluate.
    """
    ids_no_evaluate = [i for i,pos_tag in enumerate(vocabulary_labels) if pos_tag in tags_no_evaluate]

    def f(v):
        return v in ids_no_evaluate
    mask = np.vectorize(f)(y_true)
    mask = np.logical_or(mask, np.vectorize(f)(y_pred))

    return mask


def _mask_tags_no_evaluate(y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], vocabulary_labels : np.array, 
                           x : np.array = None):
    """Mask out the tags to not evaluate.

    Parameters
    ----------
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.
    x : np.array, optional
        Additional np.array, with same shape of `y_true` and `y_pred`, that we want to mask out.
        By default None

    Returns
    -------
    y_true : np.array
        Masked `y_true`. It's a monodimensional array.
    y_pred : np.array
        Masked `y_pred`. It's a monodimensional array.
    x : np.array
        Masked `x`. It's a monodimensional array.
        This is returned only if `x` is not None.
    """
    mask = _compute_mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)

    y_true = y_true[~mask]
    y_pred = y_pred[~mask]

    if x is not None:
        x = x[~mask]
        return y_true, y_pred, x

    return y_true, y_pred


def compute_f1_score(y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], vocabulary_labels : np.array) -> float:
    """Compute the f1 macro, without considering the specified POS tags in the evaluation.

    Typically, the tags to not evaluate are the punctuation tags and the padding tag.

    Parameters
    ----------
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.

    Returns
    ----------
    f1_score : float
    """
    y_true, y_pred = _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)

    return f1_score(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), average='macro')


def compute_class_report(y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], vocabulary_labels : np.array, 
                        show : bool = False, k = None, plot : bool = False):
    """Compute the classification report, without considering the specified POS tags in the evaluation.

    Parameters
    ----------
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.
    show : bool, optional
        Whether to print or not the classification report, by default False
    plot : bool, optional
        Whether to plot or not the POS tags with their scores (i.e. precision, recall, f1 score, support), sorted according 
        to the f1 score in an ascending order.
        By default False.
    k : int, optional
        If specified, only the first `k` worst predicted POS tags are plotted, by default None.
        This argument is effective only if `plot` is True.

    Returns
    -------
    class_report : dict
        The classification report dictionary. For each class, its scores are contained: precision, recall, f1-score, support.
        In addition, also the weigthed overall scores are contained.    
    """
    y_true, y_pred = _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)

    tags_to_evaluate = [tag for i, tag in enumerate(vocabulary_labels) if np.any(y_true==i) or np.any(y_pred==i)]
    # Classification report dictionary
    class_report = classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=True)

    if show:
        print(classification_report(y_true=np.ravel(y_true), y_pred=np.ravel(y_pred), 
                                    target_names=tags_to_evaluate, zero_division=0, output_dict=False))

    if plot:
        # Classification report dictionary, containing noly the entries related to the classes (and not the overall weigthed 
        # measures). The classes entries are sorted according to the f1-score, in an ascending order.
        class_report_sorted = sorted([(class_report[label]['f1-score'], label) 
                                    for label in class_report.keys() if label not in ['accuracy','macro avg','weighted avg'] ])
        class_report_sorted = class_report_sorted
        class_report_sorted = OrderedDict({tag:class_report[tag] for (f1_score, tag) in class_report_sorted})

        # Number of classes to plot: the first `k` worst classes are plotted
        k = len(vocabulary_labels) if k is None else k
        # Classes to plot
        tags_to_plot = [tag for i, tag in enumerate(class_report_sorted) if i<k]

        plt.figure(figsize=(12,12))
        x_axis = 2*np.arange(len(tags_to_plot))
        plt.bar(x_axis-0.5, [class_report_sorted[tag]['precision'] for tag in tags_to_plot], label='precision', width=0.5)
        b = plt.bar(x_axis, [class_report_sorted[tag]['recall'] for tag in tags_to_plot], label='recall', width=0.5)
        plt.bar(x_axis+0.5, [class_report_sorted[tag]['f1-score'] for tag in tags_to_plot], label='f1-score', width=0.5)
        plt.bar_label(b, labels=[class_report_sorted[tag]['support'] for tag in tags_to_plot], label_type='center')
        plt.xticks(x_axis, list(tags_to_plot))
        plt.grid(axis='y')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

    return class_report


def wrongly_classified_tokens_analysis(x : np.array, y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], 
                                       vocabulary_labels : np.array, vocabulary : np.array, k : int = 20,
                                       plot : bool = False) -> Tuple[OrderedDict, pd.DataFrame]:
    """Analyze the worst classified tokens.

    The analysis is based on the absolute number of misclassified instances of each token (in this count, the POS tags to not
    evaluate, i.e. the punctuation, are not counted).

    Parameters
    ----------
    x : np.array
        Two-dimensional array, containing the tokens ids (axis 0 : sentences; axis 1 : tokens)
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.
    vocabulary : np.array
        Array of strings, representing the mapping from integer ids to words.
    k : int, optional
        Number of tokens to analyze, by default 20.
        Basically, the worst `k` classified tokens are analyzed.
    plot : bool, optional
        Whether to plot or not the worst classified tokens, by default False.
        For each token, also the proportion of different true POS tags of the instances of that token is shown.

    Returns
    -------
    wrong_tokens_dict : OrderedDict 
        It contains the worst `k` classified tokens (represented as string), with also the corresponding number of 
        misclassified instances.
    wrong_tokens_tags_df : pd.DataFrame
        Along the rows there are the worst `k` classified tokens and along the columns there are the POS tags. For each 
        misclassified token, we store the number of true POS tags of the misclassified instances of that token.
    """
    y_true, y_pred, x = _mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels, x)

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Mask out the correctly predicted classes, i.e. keep only the misclassified tokens.
    wrong_class_mask = y_true!=y_pred 
    y_true = y_true[wrong_class_mask]
    y_pred = y_pred[wrong_class_mask]
    x = x[wrong_class_mask]

    # Integers ids of the misclassified tokens
    wrong_tokens_ids = np.array(list(Counter(x).keys()))
    # Number of misclassified instances for each misclassified token
    wrong_tokens_counts = list(Counter(x).values())
    # Integers ids of the misclassified tokens, sorted in a descending order by the number of misclassified instances
    wrong_tokens_ids = wrong_tokens_ids[np.argsort(wrong_tokens_counts)][::-1]
    # Number of misclassified instances for each misclassified token, sorted in a descending order
    wrong_tokens_counts = sorted(wrong_tokens_counts)[::-1]
    # Misclassified tokens (string representation), sorted in a descending order by the number of misclassified instances
    wrong_tokens = vocabulary[wrong_tokens_ids]

    # Keep only the worst `k` misclassified tokens
    wrong_tokens_ids, wrong_tokens, wrong_tokens_counts = wrong_tokens_ids[:k], wrong_tokens[:k], wrong_tokens_counts[:k]

    # Create the DataFrame `wrong_tokens_tags_df`: along the rows there are the misclassified tokens and along the columns
    # there are the POS tags. For each misclassified token, we store the number of true POS tags of the misclassified
    # instances of that token.
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

    if plot:
        plt.figure(figsize=(11,11))
        bottom = [0 for _ in wrong_tokens_tags_df.index]
        for tag in wrong_tokens_tags_df.columns:
            b = plt.bar(x=wrong_tokens_tags_df.index, height=wrong_tokens_tags_df[tag], bottom=bottom)
            bottom += wrong_tokens_tags_df[tag]
            plt.bar_label(b, labels=[tag if wrong_tokens_tags_df[tag][i]>0 else '' 
                                     for i in range(wrong_tokens_tags_df.shape[0])], label_type='center')
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.set_ylim([0, sum(wrong_tokens_tags_df.iloc[0])+1])
        plt.grid(axis='y');
        plt.ylabel('Number of misclassified instances')
        plt.xlabel('Misclassified tokens')
        plt.title(f'Worst {k} misclassified tokens')
        plt.show()

    wrong_tokens_dict = OrderedDict(zip(wrong_tokens, wrong_tokens_counts))

    return wrong_tokens_dict, wrong_tokens_tags_df


def _create_output_colored_string(sentence : List[str], sentence_tagsNoEvaluate_mask : np.array, 
                                  sentence_mask_wrongClass : np.array, sentence_y_true : np.array, 
                                  sentence_y_pred : np.array, vocabulary_labels : np.array) -> str:
    """Create the output string corresponding to the given sentence.

    The words are colored in different ways: green if correct word, red if misclassified word, white if non-evaluated word
    (i.e. punctuation).

    Parameters
    ----------
    sentence : List[str]
        Sentence to print out
    sentence_tagsNoEvaluate_mask : np.array
        Monodimensional boolean array localizing the tokens to not evaluate in the sentence..
    sentence_mask_wrongClass : np.array
        Monodimensional boolean array localizing the misclassified tokens in the sentence.
    sentence_y_true : np.array
        Monodimensional array containing the true POS tags of the tokens in the sentence.
    sentence_y_pred : np.array
        Monodimensional array containing the predicted POS tags of the tokens in the sentence.
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.

    Returns
    -------
    output_string : str
    """


    right_color = Fore.GREEN  # Color for the correctly classified word
    noEvaluate_color = Style.RESET_ALL  # Color for the non-evaluation word
    wrong_color = Fore.RED  # Color for the misclassified word

    output_string = ''
    for i, word in enumerate(sentence):
        additional_str = ''
        if sentence_tagsNoEvaluate_mask[i]:
            color = noEvaluate_color
        elif sentence_mask_wrongClass[i]:
            color = wrong_color
            true_tag, wrong_tag = vocabulary_labels[sentence_y_true[i]], vocabulary_labels[sentence_y_pred[i]]
            additional_str = f'[{true_tag}/{wrong_tag}]'
        else:
            color = right_color
        output_string += f'{color}{word}' + additional_str + ' '

    return output_string


def wrongly_classified_sentences_analysis(x : np.array, y_true : np.array, y_pred : np.array, tags_no_evaluate : List[str], 
                                          vocabulary_labels : np.array, vocabulary : np.array, 
                                          use_absolute_error : bool = True, k : int = 20, show : bool = False):
    """Analyze the worst classified sentences.

    The analysis is based on the relative error of each sentence, which is the number of misclassified words divided by the 
    number of total words (in this counts, the POS tags to not evaluate, i.e. the punctuation, are not counted).

    Parameters
    ----------
    x : np.array
        Two-dimensional array, containing the tokens ids (axis 0 : sentences; axis 1 : tokens)
    y_true : np.array
        Two-dimensional array, containing the true POS tags (axis 0 : sentences; axis 1 : tokens)
    y_pred : np.array
        Two-dimensional array, containing the predicted POS tags (axis 0 : sentences; axis 1 : tokens)
    punctuation_integers : list of str
        List of POS tags to not consider in the evaluation
    vocabulary_labels : np.array
        Array of strings, representing the mapping from integer ids to POS tags.
    vocabulary : np.array
        Array of strings, representing the mapping from integer ids to words.
    k : int, optional
        Number of sentences to analyze, by default 20.
        Basically, the worst `k` classified sentences are analyzed.
    show : bool, optional
        Whether to print or not the sentences, by default False

    Returns
    -------
    worst_sentences_dict : OrderedDict
        It contains the sentences (represented as integer id), with also the corresponding relative error, sorted by relative
        error.
    """
    # Mask localizing the tags to not evaluate
    tagsNoEvaluate_mask = _compute_mask_tags_no_evaluate(y_true, y_pred, tags_no_evaluate, vocabulary_labels)
    # Mask localizing the misclassified tokens
    mask_wrongClass = y_true!=y_pred 
    # Overall mask: tokens to evaluate and which are misclassified
    mask_wrongClass = np.logical_and(~tagsNoEvaluate_mask, mask_wrongClass)

    # For each sentence, there is the number of misclassified tokens
    sentences_errors_counts = np.sum(mask_wrongClass, axis=1)
    if use_absolute_error:
        # Indices of the sentences sorted by error counts in descending order
        sentences_indeces_sorted = np.argsort(sentences_errors_counts)[::-1]
        # Error counts of the sentences, sorted in descending order
        sentences_errors_counts = sorted(sentences_errors_counts)[::-1]
        sentences_errors = sentences_errors_counts
    else:
        # For each sentence, there is its length
        sentences_lengths = np.sum(~tagsNoEvaluate_mask, axis=1)
        # For each sentence, there is its relative error
        sentences_errors_relativeErrors = sentences_errors_counts/sentences_lengths
        # Indices of the sentences sorted by relative error in descending order
        sentences_indeces_sorted = np.argsort(sentences_errors_relativeErrors)[::-1]
        # Relative errors of the sentences, sorted in descending order
        sentences_errors_relativeErrors = sorted(sentences_errors_relativeErrors)[::-1]
        sentences_errors = sentences_errors_relativeErrors

    # Ordered dict which contains the sentences (represented as integer id), with also the corresponding error, 
    # sorted by relative error
    worst_sentences_dict = OrderedDict(zip(sentences_indeces_sorted, sentences_errors))

    if show:
        # Keep the worst `k` sentences
        sentences_indeces_sorted, sentences_errors = sentences_indeces_sorted[:k], sentences_errors[:k]

        # Print the legend
        right_color = Fore.GREEN  # Color for the correctly classified word
        noEvaluate_color = Style.RESET_ALL  # Color for the non-evaluation word
        wrong_color = Fore.RED  # Color for the misclassified word
        print('LEGEND')
        print(f'\t {right_color}word{Style.RESET_ALL}: correctly classified word')
        print(f'\t {wrong_color}word[TRUE_TAG/WRONG_TAG]{Style.RESET_ALL}: misclassified word')
        print(f'\t {noEvaluate_color}word{Style.RESET_ALL}: non-evaluated word (i.e. punctuation)')
        print()

        # Iterate across all sentences
        for i in range(k):
            # Print each sentence, underlying the misclassified tokens
            sentence_index = sentences_indeces_sorted[i]
            error = sentences_errors[i]
            print(f'{i+1}) Sentence index {sentence_index}')
            error_prefix_string = 'Error count' if use_absolute_error else 'Relative error'
            error_formatted = f'{error:.2f}' if isinstance(error, np.float64) else str(error)
            print(f'{error_prefix_string}: {error_formatted}')
            sentence = vocabulary[x[sentence_index]]
            sentence = [token for token in sentence if token!='']
            sentence_string = _create_output_colored_string(sentence, 
                                                            sentence_tagsNoEvaluate_mask=tagsNoEvaluate_mask[sentence_index], 
                                                            sentence_mask_wrongClass=mask_wrongClass[sentence_index], 
                                                            sentence_y_pred=y_pred[sentence_index],
                                                            sentence_y_true=y_true[sentence_index], 
                                                            vocabulary_labels=vocabulary_labels)
            print(sentence_string)
            print()

    return worst_sentences_dict
