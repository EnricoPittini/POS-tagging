# POS tagging 

POS tagging through Recurrent Neural Architectures.

Part-Of-Speech Tagging (POS Tagging)} is the process consisting in mapping each word in a text to its corresponding part of speech (e.g.: adjective, noun), in relation to its context.

The following dataset has been used: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip.

The $100$ dimensional GloVe embedding model has been used: https://nlp.stanford.edu/data/glove.6B.zip

The OOV words have been handled by performing the GloVe training procedure.

The following is the list of tried models.
- Bidirectional LSTM plus final Dense layer.
- Bidirectional GRU plus final Dense layer.
- Two Bidirectional LSTM plus final Dense layer.
- Bidirectional LSTM plus two final Dense layers.

There are two python notebooks, which must be read and run in this order.
1. The first is `1) oov handling.ipynb`, containing the code for handling the OOV words.
2. The second is `2) pos tagging.ipynb`, containing the code for solving the POS tagging task.

## Dependencies
- [NumPy](https://pypi.org/project/numpy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Pandas](https://pypi.org/project/pandas/)
- [Tensorflow](https://pypi.org/project/tensorflow/)
- [PyTorch](https://pypi.org/project/torch/)

## Repository structure

    .
    ├── dataset    # It contains the dataset files
    ├── glove_pretrained    # It contains the GloVe embedding models                           
    ├── models     # It contains the models                           
    ├── utils    # It contains the python files with useful functions
    ├── weigths       # It contains the models weigths
    ├── assignment.ipynb     # Task description
    ├── 1) oov handling.ipynb   
    ├── 2) pos tagging.ipynb 
    ├── .gitignore
    ├── LICENSE
    ├── report.pdf                          # Report of the assignment
    └── README.md

## Versioning

Git is used for versioning.

## Group members

|  Name           |  Surname  |     Email                           |    Username                                             |
| :-------------: | :-------: | :---------------------------------: | :-----------------------------------------------------: |
| Samuele         | Bortolato  | `samuele.bortolato@studio.unibo.it` | [_Sam_](https://github.com/samuele-bortolato)               |
| Antonio         | Politano  | `antonio.politano2@studio.unibo.it` | [_S1082351_](https://github.com/S1082351)               |
| Enrico          | Pittini   | `enrico.pittini@studio.unibo.it`    | [_EnricoPittini_](https://github.com/EnricoPittini)     |
| Riccardo        | Spolaor   | `riccardo.spolaor@studio.unibo.it`  | [_RiccardoSpolaor_](https://github.com/RiccardoSpolaor) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
