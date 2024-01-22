import json
import logging
import os
from pathlib import Path

import torch
from sklearn.feature_extraction.text import CountVectorizer


def get_logger(name=__name__, log_file=None, log_level=logging.INFO):
    """default log level DEBUG"""
    logger = logging.getLogger(name)
    fmt = "%(asctime)s %(filename)s %(lineno)d: %(message)s"
    datefmt = "%y-%m-%d %H:%M"
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, "w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    logger.setLevel(log_level)
    return logger


logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG)


class Vocabulary:
    """
    This vocabulary is for classifiers and pure auto regression generators, where <unk> is not needed.
    WAE vocabulary is based on this vocabulary but slightly more clear logic. Only one real difference is WAE adds <unk> special token.
    """

    def __init__(self, vocabulary, max_len=0):
        if "<pad>" not in vocabulary:
            vocabulary = {
                character: value + 1 for character, value in vocabulary.items()
            }
            vocabulary["<pad>"] = 0
        if "<start>" not in vocabulary:
            vocabulary["<start>"] = len(vocabulary)
        if "<end>" not in vocabulary:
            vocabulary["<end>"] = len(vocabulary)
        self.inv_vocabulary = {v: k for k, v in vocabulary.items()}
        self.vocabulary = vocabulary
        self.max_len = max_len
        logger.info("vocabulary length %s", len(self.vocabulary))

    def letter_to_index(self, letter):
        return self.vocabulary[letter]

    def index_to_letter(self, index):
        return self.inv_vocabulary[index]

    def tensor_to_seq(self, tensor, debug=False):
        if len(tensor.shape) == 0:
            return ""
        items = []
        for element in tensor:
            new_char = self.inv_vocabulary[element.item()]
            if not debug:
                if new_char == "<start>" or new_char == "<pad>" or new_char == "<end>":
                    break
            items.append(new_char)
        seq = "".join(items)
        return seq

    def __len__(self):
        return len(self.vocabulary)

    def seq_to_tensor(self, seq):
        tensor = torch.zeros(len(seq) + 2, dtype=torch.long)
        tensor[0] = torch.tensor(self.vocabulary["<start>"])
        for i, letter in enumerate(seq):
            tensor[i + 1] = torch.tensor(self.vocabulary[letter])
        tensor[-1] = torch.tensor(self.vocabulary["<end>"])
        return tensor
    
    def seq_to_tensor_no_start(self, seq):
        if self.max_len > 0:
            tensor = torch.zeros(self.max_len+1, dtype=torch.long)
        else:
            tensor = torch.zeros(len(seq) + 1, dtype=torch.long)
        for i, letter in enumerate(seq):
            tensor[i] = torch.tensor(self.vocabulary[letter])
        tensor[len(seq)] = torch.tensor(self.vocabulary["<end>"])
        return tensor

    def save_vocab(self, save_file):
        """ """
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=4)

    @classmethod
    def get_vocabulary_from_sequences(cls, seqs):
        """
        For only upper natural aa sequences, orig_vocabulary index starts from 0 for A, 1 for C, 2 for D, etc.
        """
        count_vect = CountVectorizer(lowercase=False, analyzer="char")
        count_vect.fit(seqs)
        orig_vocabulary = count_vect.vocabulary_
        char_voc_ = {}
        for k, v in orig_vocabulary.items():
            char_voc_[k] = v + 1
        char_voc_["<pad>"] = 0
        return Vocabulary(char_voc_)

    @classmethod
    def load_vocab_file(cls, vocab_file, max_len=0):
        """Loads json file"""
        Path(vocab_file).parent.mkdir(exist_ok=True, parents=True)
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        return Vocabulary(vocab_dict, max_len=max_len)


if __name__ == "__main__":
    seqs = ["ABCDCAD", "EDADAa"]
    count_vect = CountVectorizer(lowercase=False, analyzer="char")
    count_vect.fit(seqs)
    char_voc = count_vect.vocabulary_
    # <class 'dict'>
    print(type(char_voc))
    # {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'a': 5}
    print(char_voc)
