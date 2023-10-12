"""
This is a boilerplate pipeline 'finetuned_llm_classifier'
generated using Kedro 0.18.13
"""
import logging
import re

import lightning as L
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from kedro_azureml.distributed import distributed_job
from kedro_azureml.distributed.config import Framework
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from mlops_template.pipelines.log_sklearn_metrics.nodes import (
    format_classification_report,
)

from .stopwords import STOPWORDS

log = logging.getLogger(__name__)


class TokenizedDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased", return_dict=False
        )

    def __len__(self):
        return len(self.df)

    def clean_text(self, text: str, stopwords: list[str] = STOPWORDS) -> str:
        """Clean raw text string.
        Args:
            text (str): Raw text to clean.
            stopwords (List, optional): list of words to filter out.
            Defaults to STOPWORDS.

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

    def tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        # Tokenize the text using the tokenizer
        inputs = self.tokenizer(
            text,
            return_tensors="np",  # Return PyTorch tensors
            padding="longest",  # Padding to the longest sequence in the batch
            truncation=True,  # Truncate sequences if they exceed max length
        )

        return inputs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text: str = row["title"] + " " + row["description"]
        text = self.clean_text(text)
        tokenized_text = self.tokenize_text(text)
        target = torch.tensor(row["tag"], dtype=torch.int64)  # Convert target to tensor
        return {
            "input_ids": tokenized_text["input_ids"].flatten(),  # 1d
            "attention_masks": tokenized_text["attention_mask"].flatten(),  # 1d
            "targets": target,  # scalar
        }


class TokenizedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_val_df,
        test_df,
        val_proportion=0.2,
        batch_size=32,
    ):
        super().__init__()
        self.train_val_df = train_val_df
        self.test_df = test_df
        self.val_proportion = val_proportion
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_df, self.val_df = train_test_split(
            self.train_val_df,
            test_size=self.val_proportion,
            stratify=self.train_val_df["tag"],
        )

        self.train_dataset = TokenizedDataset(self.train_df)
        self.val_dataset = TokenizedDataset(self.val_df)
        self.test_dataset = TokenizedDataset(self.test_df)

    def pad_array(self, arr, dtype=np.int32):
        max_len = max(len(row) for row in arr)
        padded_arr = np.zeros((len(arr), max_len), dtype=dtype)
        for i, row in enumerate(arr):
            padded_arr[i][: len(row)] = row
        return padded_arr

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_masks"] for item in batch]
        targets = [item["targets"] for item in batch]

        padded_ids = self.pad_array(input_ids)
        padded_masks = self.pad_array(attention_masks)
        targets = torch.stack(targets)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_batch = {}
        for key, array, dtype in [
            ("input_ids", padded_ids, torch.int32),
            ("attention_mask", padded_masks, torch.int32),
            ("targets", targets, torch.int64),
        ]:
            tensor_batch[key] = torch.as_tensor(array, dtype=dtype, device=device)
        return tensor_batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )


class FinetunedLlmModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.5,
        lr: float = 1e-4,
        lr_factor: float = 0.8,
        lr_patience: int = 3,
        batch_size: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.batch_size = batch_size
        self.llm = BertModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", return_dict=False
        )
        self.embedding_dim = self.llm.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(self.embedding_dim, self.num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

    def forward(self, batch):
        seq, pool = self.llm(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
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
        onehot_targets = F.one_hot(
            batch["targets"], num_classes=self.num_classes
        ).float()
        loss = self.loss_fn(z, onehot_targets)
        self.log("val_loss", loss)
        self.accuracy(z, batch["targets"])
        self.f1_macro(z, batch["targets"])
        self.log("val_acc", self.accuracy)
        self.log("val_f1_macro", self.f1_macro)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        z = self(batch)
        onehot_targets = F.one_hot(
            batch["targets"], num_classes=self.num_classes
        ).float()
        loss = self.loss_fn(z, onehot_targets)

        if batch_idx == 0:
            self.test_step_outputs = []

        predicted_probs = F.softmax(z, dim=1)
        predicted_labels = torch.argmax(predicted_probs, dim=1)
        predicted_labels = predicted_labels.cpu().numpy()
        output = {
            "test_loss": loss,
            "y_true": batch["targets"].cpu().numpy(),
            "y_pred": predicted_labels,
        }

        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        all_y_true = np.concatenate(
            [output["y_true"] for output in self.test_step_outputs]
        )
        all_y_pred = np.concatenate(
            [output["y_pred"] for output in self.test_step_outputs]
        )
        self.classification_report = classification_report(
            all_y_true, all_y_pred, output_dict=True
        )

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


@distributed_job(Framework.PyTorch, num_nodes="params:finetuned_llm_config.num_nodes")
def train_llm_classifier(train_val_df, test_df, config, label_encoder_mapping):
    num_classes = len(label_encoder_mapping)
    batch_size = int(config["batch_size"])
    dropout_p = float(config["dropout_p"])
    lr = float(config["lr"])
    lr_factor = float(config["lr_factor"])
    lr_patience = int(config["lr_patience"])
    num_epochs = int(config["num_epochs"])
    checkpoint_dir = config["checkpoint_dir"]
    resume_checkpoint_from = config["resume_checkpoint_from"]
    num_nodes = int(config["num_nodes"])  # noqa: F841

    datamodule = TokenizedDataModule(
        train_val_df=train_val_df,
        test_df=test_df,
        val_proportion=0.2,
        batch_size=config["batch_size"],
    )

    if resume_checkpoint_from:
        model = FinetunedLlmModule.load_from_checkpoint(
            resume_checkpoint_from,
            num_classes=num_classes,
            dropout_p=dropout_p,
            lr=lr,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            batch_size=batch_size,
        )

    else:
        model = FinetunedLlmModule(
            num_classes=num_classes,
            dropout_p=dropout_p,
            lr=lr,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            batch_size=batch_size,
        )

    mlflow_tracking_uri = mlflow.get_tracking_uri()
    run_id = mlflow.active_run().info.run_id

    mlflow_logger = MLFlowLogger(
        run_id=run_id,
        tracking_uri=mlflow_tracking_uri,
    )

    trainer = L.Trainer(
        default_root_dir=checkpoint_dir,
        enable_progress_bar=True,
        max_epochs=num_epochs,
        logger=mlflow_logger,
        # callbacks=[ProgressBar()],
        num_nodes=num_nodes,
        log_every_n_steps=5,
    )
    trainer.fit(model, datamodule)
    return model, datamodule


def evaluate_model(model, datamodule):
    trainer = L.Trainer(
        log_every_n_steps=5,
    )
    trainer.test(model, datamodule)
    return model, format_classification_report(model.classification_report)


# df = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv") # noqa: E501
# test_df = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv") # noqa: E501

# df["tag"] = 0
# test_df["tag"] = 2
# bs = 2
# dm = TokenizedDataModule(df, test_df, batch_size=bs)
# dm.setup()
# iterator = iter(dm.train_dataloader())
# next(iterator)
