import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from os import path

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import ConversationDataset, collate_fn
from model.entrainment import EntrainmentModel

if __name__ == "__main__":
    tail_length = -1
    history = "self"
    attention = False
    corpus_dir = "/home/mmcneil/datasets/fisher_corpus"
    epochs = 10
    input_features = [
        "pitch_mean_norm",
        "pitch_range_norm",
        "intensity_mean_norm",
        "jitter_norm",
        "shimmer_norm",
        "nhr_norm",
        "rate_norm",
    ]
    output_features = input_features  # ["intensity_mean_norm"]  # input_features
    feature_dir = "data/fisher"

    feature_idx = dict((v, k) for (k, v) in enumerate(input_features))

    df_turns = pd.read_csv(path.join(feature_dir, "turns_norm.csv"))
    df_turns = df_turns.set_index("ses_id")

    idxs, test_idx = train_test_split(
        list(df_turns.index.unique()), train_size=0.90, random_state=9001
    )
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=9001)

    batch_size = 128

    train_dataset = ConversationDataset(
        df_turns,
        idxs,
        input_features,
        output_features,
        word_embeddings=True,
        embeddings_dir="/home/mmcneil/datasets/fisher_corpus/turns_glove",
    )
    test_dataset = ConversationDataset(
        df_turns,
        test_idx,
        input_features,
        output_features,
        word_embeddings=True,
        embeddings_dir="/home/mmcneil/datasets/fisher_corpus/turns_glove",
    )

    val_dataset = ConversationDataset(
        df_turns,
        val_idx,
        input_features,
        output_features,
        word_embeddings=True,
        embeddings_dir="/home/mmcneil/datasets/fisher_corpus/turns_glove",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=8,
    #     persistent_workers=True,
    #     collate_fn=collate_fn,
    #     pin_memory=True,
    # )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = EntrainmentModel(
        lr=0.0001,
        feature_dim=7,
        feature_encoder_out_dim=256,
        feature_encoder_dropout=0.5,
        feature_attention_dim=128,
        decoder_out_dim=256,
        evaluation_mode="us",
        evaluation_loss_mode="us",
        training_mode="us",
        training_loss_mode="us",
        teacher_forcing_mode="us",
        embeddings=True,
        embedding_dim=300,
        embedding_encoder_out_dim=256,
        embedding_encoder_dropout=0.5,
        embedding_attention_dim=128,
        teacher_forcing=0.5,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        precision=16,
        max_epochs=-1,
        strategy="ddp_find_unused_parameters_false",
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
