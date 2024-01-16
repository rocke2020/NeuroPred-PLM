from functools import cache
import torch, os, sys

sys.path.append(os.path.abspath("."))
import esm
from torch.nn.utils.rnn import pad_sequence
import time
from utils_comm.log_util import logger
from utils_comm.train_util import get_device
import torch.nn as nn
from typing import Sequence, List, Tuple
from copy import deepcopy
from torch import Tensor
from .vocabulary import Vocabulary


class Dataset(torch.utils.data.Dataset):
    """ """

    def __init__(
        self, df, vocabulary: Vocabulary, category_key="", seq_col_name="Sequence"
    ):
        self.vocabulary = vocabulary
        self.df = df
        self.category_key = category_key
        self.seq_col_name = seq_col_name

    def __getitem__(self, i):
        seq_str = self.df.iloc[i][self.seq_col_name]
        ## debug
        # try:
        #     tensor = self.vocabulary.seq_to_tensor(seq_str)
        # except Exception as identifier:
        #     logger.error(f'error in seq_to_tensor {self.df["Sequence"].head()}')
        #     logger.error(f'error in seq_to_tensor {seq_str}')
        #     raise(identifier)
        tensor = self.vocabulary.seq_to_tensor(seq_str)
        if self.category_key:
            category = self.df.iloc[i][self.category_key]
            return tensor, category
        else:
            return tensor

    def __len__(self):
        return len(self.df)


class DatasetPredict(torch.utils.data.Dataset):
    """Input seqs and without category"""

    def __init__(self, seqs: list[str], vocabulary: Vocabulary):
        self.vocabulary = vocabulary
        self.seqs = seqs

    def __getitem__(self, i):
        seq_str = self.seqs[i]
        tensor = self.vocabulary.seq_to_tensor(seq_str)
        return tensor

    def __len__(self):
        return len(self.seqs)


def collate_fn_rnn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    arr_seq = [element[0] for element in arr]
    cat_seq = [element[1] for element in arr]
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr_seq, enforce_sorted=False)
    seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
        packed_seq, batch_first=True
    )
    return seq_unpacked, lens_unpacked, torch.tensor(cat_seq)


def collate_fn_cnn(arr, total_length=52):
    """Function to take a list of encoded sequences and turn them into a batch, vocabulary adds start and end."""
    arr_seq = [element[0] for element in arr]
    cat_seq = [element[1] for element in arr]
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr_seq, enforce_sorted=False)
    seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
        packed_seq, batch_first=True, total_length=total_length
    )
    return seq_unpacked, lens_unpacked, torch.tensor(cat_seq)


def collate_fn_pretrain_rnn(arr):
    """ """
    seq_emb_tensors = [element[0] for element in arr]
    lengths = [element[1] for element in arr]
    categories = [element[2] for element in arr]
    padded_seq = pad_sequence(seq_emb_tensors, batch_first=True)
    return padded_seq, lengths, categories


def collate_fn_pretrain_cnn(arr):
    """ """
    padded_seq = [element[0] for element in arr]
    lengths = [element[1] for element in arr]
    categories = [element[2] for element in arr]
    return padded_seq, lengths, categories


def collate_fn_no_category(arr) -> Tuple[Tensor, Tensor]:
    """Function to take a list of encoded sequences and turn them into a batch
    Args:
        arr: list[tensor], inside tensor shape: seq-len * dim-size
    Returns:
        seq_unpacked, lens_unpacked
    """
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr, enforce_sorted=False)
    padded_seq = torch.nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
    return padded_seq


def collate_fn_cnn_no_category(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    padded_seq = torch.stack(arr).unsqueeze(dim=1)
    # logger.info(f'padded_seq.shape {padded_seq.shape}')
    return padded_seq


class Dataset_PretrainedEmbedding(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.cache_file = ""

    def __len__(self):
        return len(self.dataset)

    def load_cached_dataset(self):
        logger.info(f"Starts to load cached dataset from {self.cache_file}")
        self.dataset = torch.load(self.cache_file)
        logger.info(
            f"Already loads the cached dataset from {self.cache_file}, len {len(self.dataset)}"
        )


class ESMBatchTokenizer(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = 0, max_len=50):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.max_len = max_len

    def __call__(self, seq_str_list: Sequence[str]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(seq_str_list)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list
            ]
        if isinstance(self.max_len, int) and self.max_len > 0:
            max_len = self.max_len
        else:
            max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_encoded) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return tokens


class Dataset_ESMEmbedding(Dataset_PretrainedEmbedding):
    def __init__(
        self,
        input_df,
        convert_batch_size=100,
        category="",
        device=None,
        read_cache=True,
        cache_file=None,
        max_len=None,
    ):
        self.category_key = category
        self.dataset = []
        self.cache_file = cache_file
        self.read_cache = read_cache
        self.input_df = input_df
        self.convert_batch_size = convert_batch_size
        self.max_len = max_len
        if self.read_cache and self.cache_file.is_file():
            self.load_cached_dataset()
        else:
            if device:
                self.device = device
            else:
                self.device = get_device()
            self.create_dataset()

    def __getitem__(self, i):
        tensor, length, category = self.dataset[i]
        _tensor = deepcopy(tensor)
        _length = deepcopy(length)
        if self.category_key:
            _category = deepcopy(category)
            return _tensor, _length, _category
        else:
            return _tensor, _length

    def create_dataset(
        self,
    ):
        logger.info("Starts to create dataset from pretrained transformer model")
        self.batch_tokenizer, self.model, self.esm_layer_num = get_esm_model(
            max_len=self.max_len
        )
        self.model = self.model.to(self.device)
        t0 = time.time()
        with torch.no_grad():
            inputs = []
            categories = []
            for i in range(len(self.input_df)):
                seq_str = self.input_df.iloc[i]["Sequence"]
                if self.max_len and len(seq_str) > self.max_len:
                    continue
                category = self.input_df.iloc[i][self.category_key]
                inputs.append(seq_str)
                categories.append(category)
                if (i + 1) % self.convert_batch_size == 0:
                    self.add_pretrained_embeddings(inputs, categories)
                    inputs.clear()
                    categories.clear()
            if inputs:
                self.add_pretrained_embeddings(inputs, categories)
        t1 = time.time() - t0
        logger.info(f"create_dataset uses seconds {t1}")
        if self.cache_file:
            logger.info(f"Saves dataset to {self.cache_file}")
            torch.save(self.dataset, self.cache_file)

    def add_pretrained_embeddings(self, inputs, categories):
        batch_tokens = self.batch_tokenizer(inputs)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(
            batch_tokens, repr_layers=[self.esm_layer_num], return_contacts=False
        )
        token_representations = results["representations"][self.esm_layer_num]
        for i, (seq, category) in enumerate(zip(inputs, categories)):
            if self.max_len:
                self.dataset.append(
                    [token_representations[i].cpu(), len(seq), category]
                )
            else:
                self.dataset.append(
                    [
                        token_representations[i, 1 : len(seq) + 1].cpu(),
                        len(seq),
                        category,
                    ]
                )


class Dataset_ESMEmbedding_Predict(Dataset_PretrainedEmbedding):
    """
    Difference vs Dataset_ESMEmbedding is, here, no category.

    As create_dataset() is done in init(), don't inherit Dataset_ESMEmbedding to avoid Dataset_ESMEmbedding init(). So, create a basic Dataset_PretrainedEmbedding class.
    """

    def __init__(
        self,
        sequences: list[str],
        convert_batch_size=100,
        device=None,
        read_cache=True,
        cache_file=None,
        max_len=None,
    ):
        self.dataset = []
        self.cache_file = cache_file
        self.read_cache = read_cache
        self.sequences = sequences
        self.convert_batch_size = convert_batch_size
        self.max_len = max_len
        if self.read_cache and self.cache_file.is_file():
            self.load_cached_dataset()
        else:
            if device:
                self.device = device
            else:
                self.device = get_device()
            self.create_dataset()

    def create_dataset(
        self,
    ):
        logger.info("Starts to create dataset from pretrained transformer model")
        self.batch_converter, self.model, self.esm_layer_num = get_esm_model(
            max_len=self.max_len
        )
        self.model = self.model.to(self.device)
        t0 = time.time()
        with torch.no_grad():
            inputs = []
            for i, seq_str in enumerate(self.sequences):
                inputs.append(seq_str)
                if (i + 1) % self.convert_batch_size == 0:
                    self.add_pretrained_embeddings(inputs)
                    inputs.clear()
            if inputs:
                self.add_pretrained_embeddings(inputs)
        t1 = time.time() - t0
        if self.cache_file:
            torch.save(self.dataset, self.cache_file)
            logger.info(f"save dataset at {self.cache_file}")
        logger.info(f"create_dataset uses seconds {t1}")

    def add_pretrained_embeddings(self, inputs):
        batch_tokens = self.batch_converter(inputs)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(
            batch_tokens, repr_layers=[self.esm_layer_num], return_contacts=False
        )
        token_representations = results["representations"][self.esm_layer_num]
        for i, seq in enumerate(inputs):
            if self.max_len:
                self.dataset.append(token_representations[i].cpu())
            else:
                self.dataset.append(token_representations[i, 1 : len(seq) + 1].cpu())

    def __getitem__(self, i):
        tensor = self.dataset[i]
        return tensor


class Dataset_ESMEmbedding_del_max_len(Dataset_ESMEmbedding_Predict):
    """not re calculate, but just read the real length from cnn ESMEmbedding whhere max len is used"""

    def __init__(
        self, cnn_dataset: Dataset_ESMEmbedding_Predict, sequences: list[str]
    ) -> None:
        self.dataset = self.create_dataset(cnn_dataset, sequences)

    def create_dataset(self, cnn_dataset, sequences):
        """ """
        dataset = []
        for cnn_embedding, seq in zip(cnn_dataset, sequences):
            dataset.append(cnn_embedding[1 : len(seq) + 1])
        return dataset


@cache
def get_esm_model(model_level="650m", max_len=None):
    """the model saved dir: f"{torch.hub.get_dir()}/checkpoints/{fn}.pt", e.g. /home/qcdong/.cache/torch/hub"""
    if model_level == "650m":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_layer_num = 33
    elif model_level == "3b":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        esm_layer_num = 36
    batch_tokenizer = ESMBatchTokenizer(alphabet, max_len=max_len)
    model.eval()
    return batch_tokenizer, model, esm_layer_num


class Dataset_BERT_TOKENIZER(torch.utils.data.Dataset):
    def __init__(
        self, input_df, tokenizer, category_key="", read_cache=True, cache_file=None
    ):
        self.category_key = category_key
        self.dataset = []
        self.cache_file = cache_file
        self.read_cache = read_cache

        if self.read_cache and self.cache_file.is_file():
            self.load_cached_dataset()
        else:
            self.tokenizer = tokenizer
            self.create_dataset_from_bert_tokenizer(input_df)

        # logger.info('the first 3 items in dataset')
        # for i in range(3):
        #     input_id, attention_mask, length, category = self.dataset[i]
        #     logger.info(f'input_id {input_id}\nattention_mask {attention_mask}\nlength {length}, '
        #         f'category {category}')

    def __getitem__(self, i):
        # input_id, attention_mask, length, category
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    def load_cached_dataset(
        self,
    ):
        self.dataset = torch.load(self.cache_file)
        logger.info(f"Already loads the cached dataset from {self.cache_file}")

    def create_dataset_from_bert_tokenizer(self, input_df):
        for i in range(len(input_df)):
            seq_str = input_df.iloc[i]["Sequence"]
            category = input_df.iloc[i][self.category_key]
            sequence = " ".join(list(seq_str))
            length = len(seq_str) + 2
            encoded_input = self.tokenizer(sequence, return_tensors="pt")
            input_id = encoded_input.input_ids[0]
            attention_mask = encoded_input.attention_mask[0]
            self.dataset.append([input_id, attention_mask, length, category])

        if self.cache_file:
            torch.save(self.dataset, self.cache_file)


def collate_fn_bert(batches, pad_id=0):
    """
    Returns:
        input_ids(tensor), attention_masks(tensor), lengths(list[int]), categories(list[int])
    """
    input_ids = [element[0] for element in batches]
    attention_masks = [element[1] for element in batches]
    lengths = [element[2] for element in batches]
    categories = [element[3] for element in batches]
    input_ids = nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    )
    attention_masks = nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=pad_id
    )
    return input_ids, attention_masks, lengths, categories


class Dataset_T5Embedding(Dataset_PretrainedEmbedding):
    def __init__(
        self,
        input_df,
        model_dir="/mnt/sda/models/Rostlab/Rostlab_prot_t5_xl_uniref50",
        convert_batch_size=100,
        category="",
        read_cache=True,
        cache_file=None,
        device=None,
        max_len=None,
    ):
        self.category_key = category
        self.dataset = []
        self.cache_file = cache_file
        self.read_cache = read_cache
        self.max_len = max_len
        self.model_dir = model_dir
        if self.read_cache and self.cache_file.is_file():
            self.load_cached_dataset()
        else:
            if device:
                self.device = device
            else:
                self.device = get_device()
            logger.info("Starts to create dataset from pretrained transformer model")
            self.create_dataset(input_df, convert_batch_size)

        first_log_num = 3
        first_seqs = input_df["Sequence"][:first_log_num]
        logger.info(f"log head of input df\n{first_seqs}")
        logger.info(f"the first {first_log_num} items in dataset")
        for i in range(first_log_num):
            tensor, length, category = self.dataset[i]
            logger.info(
                f"tensor {tensor}\n tenor shape {tensor.shape} length {length}, category {category}"
            )
        logger.info("len(self.dataset) %s", len(self.dataset))

    def __getitem__(self, i):
        tensor, length, category = self.dataset[i]
        _tensor = deepcopy(tensor)
        _length = deepcopy(length)
        if self.category_key:
            _category = deepcopy(category)
            return _tensor, _length, _category
        else:
            return _tensor, _length

    def create_dataset(self, input_df, convert_batch_size):
        self.tokenizer, self.model = get_t5_model(self.model_dir)
        self.model = self.model.to(self.device)
        t0 = time.time()
        with torch.no_grad():
            inputs = []
            categories = []
            lengths = []
            for i in range(len(input_df)):
                seq_str = input_df.iloc[i]["Sequence"]
                category = input_df.iloc[i][self.category_key]
                sequence = " ".join(list(seq_str))
                inputs.append(sequence)
                categories.append(category)
                lengths.append(len(seq_str))
                if (i + 1) % convert_batch_size == 0:
                    self.add_pretrained_embeddings(inputs, categories, lengths)
                    inputs.clear()
                    categories.clear()
                    lengths.clear()
            if inputs:
                self.add_pretrained_embeddings(inputs, categories, lengths)
        t1 = time.time() - t0
        logger.info(f"create_dataset uses seconds {t1}")
        if self.cache_file:
            torch.save(self.dataset, self.cache_file)

    def add_pretrained_embeddings(self, inputs, categories, lengths):
        if self.max_len:
            tokenized_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_len,
            )
        else:
            tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)
        output = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        emb = output.last_hidden_state
        for i, (seq, category, length) in enumerate(zip(inputs, categories, lengths)):
            if self.max_len:
                self.dataset.append([emb[i].cpu(), length, category])
            else:
                self.dataset.append([emb[i, : len(seq)].cpu(), length, category])


@cache
def get_t5_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = T5Model.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def test_prot(input_seqs):
    """ """
    tokenizer, model = get_t5_model(
        "/mnt/sda/models/Rostlab/Rostlab_prot_t5_xl_uniref50"
    )
    device = get_device()
    inputs = [" ".join((seq)) for seq in input_seqs]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    logger.info("tokenizer.get_vocab() %s", tokenizer.get_vocab())
    logger.info("\n%s\n%s", input_seqs, tokenized_inputs)


if __name__ == "__main__":
    seqs = [
        "ADE",
        "ADaAX",
        "RHZ",
    ]
    # test_prot(seqs)

    batch_tokenizer, model, esm_layer_num = get_esm_model(
        model_level="650m", max_len=None
    )
