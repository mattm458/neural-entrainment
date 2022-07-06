from collections import defaultdict
from os import path

import torch
from torch import nn
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    def __init__(
        self,
        df,
        idxs,
        X_features,
        target,
        mels=False,
        mels_dir=None,
        word_embeddings=False,
        embeddings_dir=None,
    ):
        super().__init__()

        if mels and mels_dir is None:
            raise Exception(
                "Mels cannot be returned without a path to a directory of extracted mels"
            )

        if word_embeddings and embeddings_dir is None:
            raise Exception(
                "Word embeddings cannot be returned without a directory of vectors"
            )

        self.mels = mels
        self.mels_dir = mels_dir

        self.word_embeddings = word_embeddings
        self.embeddings_dir = embeddings_dir

        self.df = df
        self.idxs = idxs

        self.X_features = X_features
        self.target = target

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        ses_id = self.idxs[idx]
        X_data = self.df.loc[ses_id]

        X_features = torch.FloatTensor(X_data[self.X_features].values)
        y_features = torch.FloatTensor(X_data[self.target].values)
        speaker = torch.FloatTensor(
            [[1, 0] if x == "A" else [0, 1] for x in X_data["speaker"].values]
        )

        X_features = X_features[:-1]
        X_speaker = speaker[:-1]
        y_features = y_features[1:]
        y_speaker = speaker[1:]

        output = dict()
        output["X_features"] = X_features
        output["X_speaker"] = X_speaker
        output["X_features_len"] = torch.LongTensor([len(X_features)])
        output["y_features"] = y_features
        output["y_speaker"] = y_speaker
        output["y_features_len"] = torch.LongTensor([len(y_features)])

        if self.mels:
            mels = torch.load(path.join(self.mels_dir, f"{ses_id}-mels-cat.pt"))
            mels_len = torch.load(path.join(self.mels_dir, f"{ses_id}-mels-len.pt"))

            mels_expanded = []
            start = 0
            for mel_len in mels_len:
                mels_expanded.append(mels[start : start + mel_len])
                start += mel_len

            mels = nn.utils.rnn.pad_sequence(mels_expanded, batch_first=True)
            mels = mels.reshape(-1, 80)

            output["mels"] = mels
            output["mels_len"] = mels_len

        if self.word_embeddings:
            embeddings = torch.load(
                path.join(self.embeddings_dir, f"{ses_id}-embeddings-cat.pt")
            )
            embeddings_len = torch.load(
                path.join(self.embeddings_dir, f"{ses_id}-embeddings-len.pt")
            )

            output["embeddings"] = embeddings
            output["embeddings_len"] = embeddings_len

        return output


def collate_fn(batch):
    collated = defaultdict(list)

    for x in batch:
        collated["X_features"].append(x["X_features"])
        collated["X_speaker"].append(x["X_speaker"])
        collated["X_features_len"].append(x["X_features_len"])
        collated["y_features"].append(x["y_features"])
        collated["y_speaker"].append(x["y_speaker"])
        collated["y_features_len"].append(x["y_features_len"])

        if "mels" in x:
            collated["mels"].append(x["mels"])
            collated["mels_len"].append(x["mels_len"])

        if "embeddings" in x:
            collated["embeddings"].append(x["embeddings"])
            collated["embeddings_len"].append(x["embeddings_len"])

    collated["X_features"] = nn.utils.rnn.pad_sequence(
        collated["X_features"], batch_first=True
    )
    collated["X_speaker"] = nn.utils.rnn.pad_sequence(
        collated["X_speaker"], batch_first=True
    )
    collated["X_features_len"] = torch.LongTensor(collated["X_features_len"])
    collated["y_features"] = nn.utils.rnn.pad_sequence(
        collated["y_features"], batch_first=True
    )
    collated["y_speaker"] = nn.utils.rnn.pad_sequence(
        collated["y_speaker"], batch_first=True
    )
    collated["y_features_len"] = torch.LongTensor(collated["y_features_len"])

    if "mels" in collated:
        collated["mels"] = nn.utils.rnn.pad_sequence(collated["mels"], batch_first=True)
        collated["mels_len"] = nn.utils.rnn.pad_sequence(
            collated["mels_len"], batch_first=True
        )

    if "embeddings" in collated:
        collated["embeddings"] = nn.utils.rnn.pad_sequence(
            collated["embeddings"], batch_first=True, padding_value=1
        )
        collated["embeddings_len"] = nn.utils.rnn.pad_sequence(
            collated["embeddings_len"], batch_first=True, padding_value=1
        )

    return dict(collated)
