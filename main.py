import argparse
from os import path

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import ConversationDataset, collate_fn
from model.entrainment import EntrainmentModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A model for predicting entrainment behavior in conversations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general_parser = parser.add_argument_group("General options")
    general_parser.add_argument(
        "--embeddings-dir",
        required=False,
        help="The base directory containing precomputed word embeddings from the corpus transcript.",
        type=str,
    )
    general_parser.add_argument(
        "--feature-dir",
        required=True,
        help="The base directory containing extracted features.",
        type=str,
    )

    args = parser.parse_args()

    history = "self"
    attention = False
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

    feature_idx = dict((v, k) for (k, v) in enumerate(input_features))

    df_turns = pd.read_csv(path.join(args.feature_dir, "turns_norm.csv"))
    df_turns = df_turns.set_index("ses_id")

    idxs, test_idx = train_test_split(
        list(df_turns.index.unique()), train_size=0.80, random_state=9001
    )
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=9001)

    batch_size = 128

    train_dataset = ConversationDataset(
        df_turns,
        idxs,
        input_features,
        output_features,
        word_embeddings=True,
        embeddings_dir=args.embeddings_dir,
    )

    val_dataset = ConversationDataset(
        df_turns,
        val_idx,
        input_features,
        output_features,
        word_embeddings=True,
        embeddings_dir=args.embeddings_dir,
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
