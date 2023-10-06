"""
This is a boilerplate pipeline 'finetuned_llm_classifier'
generated using Kedro 0.18.13
"""
import re

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from .stopwords import STOPWORDS


def clean_text(text: str, stopwords: list[str] = STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out. Defaults to STOPWORDS.

    Returns:
        str: cleaned text.
    """
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def tokenize(batch: pd.DataFrame) -> dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of data with the results of tokenization (`input_ids` and `attention_mask`) on the text inputs.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "allenai/scibert_scivocab_uncased", return_dict=False
    )
    encoded_inputs = tokenizer(
        batch["text"].tolist(), return_tensors="np", padding="longest"
    )
    return dict(
        ids=encoded_inputs["input_ids"],
        masks=encoded_inputs["attention_mask"],
        targets=np.array(batch["tag"]),
    )


def preprocess(df: pd.DataFrame) -> dict:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """

    df = df.copy()
    df["text"] = df["title"] + " " + df["description"]
    df["text"] = df["text"].apply(clean_text)
    outputs = tokenize(df)
    return outputs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def pad_array(arr, dtype=np.int32):
    """
    Pad a 2D NumPy array with zeros to make all rows have the same length.

    Parameters:
    - `arr` (numpy.ndarray): The input 2D array to be padded.
    - `dtype` (numpy.dtype, optional): The data type of the resulting padded array.
      Defaults to np.int32.

    Returns:
    - `padded_arr` (numpy.ndarray): A new 2D array with the same number of rows as `arr`,
      where each row is padded with zeros to match the length of the longest row.

    Example:
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    >>> padded_arr = pad_array(arr)
    >>> print(padded_arr)
    array([[1, 2, 3, 0],
           [4, 5, 0, 0],
           [6, 7, 8, 9]], dtype=int32)

    This function takes a 2D array `arr` and pads each row with zeros to match the length
    of the longest row in the input. The resulting padded array `padded_arr` is returned
    with the specified data type or the default data type np.int32.
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch):
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(
            array, dtype=dtypes[key], device=get_device()
        )
    return tensor_batch


class TokenizedDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased", return_dict=False
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # You can customize this method to return data in the desired format
        sample = self.df.iloc[idx]
        return sample


class TokenizedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_val_df,
        test_df,
        val_proportion=0.2,
        batch_size=32,
        feature_col="text",
        target_col="tag",
    ):
        super().__init__()
        self.train_val_df = train_val_df
        self.test_df = test_df
        self.val_proportion = val_proportion
        self.batch_size = batch_size
        self.feature_col = feature_col
        self.target_col = target_col

    def setup(self, stage=None):
        self.train_df, self.val_df = train_test_split(
            self.train_val_df, test_size=self.val_proportion
        )

        self.train_outputs = preprocess(self.train_df)
        self.val_outputs = preprocess(self.val_df)
        self.test_outputs = preprocess(self.test_df)

        self.train_dataset = MyDataset(self.train_outputs)
        self.val_dataset = MyDataset(self.val_outputs)
        self.test_dataset = MyDataset(self.test_outputs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )


class FinetunedLlmModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.5,
        lr: float = 1e-4,
        lr_factor: float = 0.8,
        lr_patience: int = 3,
        num_epochs: int = 10,
        batch_size: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.llm = BertModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", return_dict=False
        )
        self.embedding_dim = self.llm.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(self.embedding_dim, self.num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass")
        self.f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

    def forward(self, batch):
        seq, pool = self.llm(input_ids=batch["ids"], attention_mask=batch["masks"])
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

    def training_step(self, batch, batch_idx):
        z = self(batch)
        targets = F.one_hot(batch["targets"], num_classes=self.num_classes).float()
        loss = self.loss_fn(z, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        z = self(batch)
        targets = F.one_hot(batch["targets"], num_classes=self.num_classes).float()
        loss = self.loss_fn(z, targets)
        self.log("val_loss", loss)
        self.accuracy(z, batch["targets"])
        self.f1_macro(z, batch["targets"])
        self.log("val_acc", self.accuracy)
        self.log("val_f1_macro", self.f1_macro)

    def test_step(self, batch, batch_idx):
        z = self(batch)
        targets = F.one_hot(batch["targets"], num_classes=self.num_classes).float()
        loss = self.loss_fn(z, targets)
        self.log("test_loss", loss)
        self.accuracy(z, batch["targets"])
        self.f1_macro(z, batch["targets"])
        self.log("test_acc", self.accuracy)
        self.log("test_f1_macro", self.f1_macro)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            verbose=True,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
