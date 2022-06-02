from os import path

import pandas as pd
from torch.utils.data import DataLoader
from dataset import WindowedFeatureDataset
from sklearn.model_selection import train_test_split


def train(
    feature_dir,
    history,
    tail_length,
    batch_size,
    input_features=["intensity_mean_norm", "pitch_range_norm", "rate_norm"],
    output_features=["intensity_mean_norm"],
):
    df_turns = pd.read_csv(path.join(feature_dir, "turns_norm.csv"))
    df_tails = pd.read_csv(
        path.join(feature_dir, "tails", f"{history}-turn-{tail_length}.csv")
    )

    train_ses, test_ses = train_test_split(
        df_turns.ses_id.unique(), train_size=0.8, random_state=9001
    )
    test_ses, val_ses = train_test_split(test_ses, test_size=0.5, random_state=9001)

    train_dataset = WindowedFeatureDataset(
        df_turns,
        df_tails[df_tails.ses_id.isin(train_ses)],
        1,
        input_features,
        output_features,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    test_dataset = WindowedFeatureDataset(
        df_turns,
        df_tails[df_tails.ses_id.isin(test_ses)],
        1,
        input_features,
        output_features,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    val_dataset = WindowedFeatureDataset(
        df_turns,
        df_tails[df_tails.ses_id.isin(val_ses)],
        1,
        input_features,
        output_features,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
