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
        features,
        target,
        mels=False,
        mels_dir=None,
        word_embeddings=False,
        embeddings_dir=None,
        include_transcript=False,
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

        self.features = features
        self.target = target

        self.include_transcript = include_transcript

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        ses_id = self.idxs[idx]
        X_data = self.df.loc[ses_id]

        features = torch.FloatTensor(X_data[self.features].values)

        first_speaker = X_data.iloc[0].speaker
        speaker = torch.FloatTensor(
            [[1, 0] if x == first_speaker else [0, 1] for x in X_data.speaker.values]
        )

        output = dict()
        output["features"] = features
        output["speaker"] = speaker
        output["features_len"] = torch.LongTensor([len(features)])

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

        if self.include_transcript:
            output["transcript"] = X_data.transcript.tolist()

        return output


def collate_fn(batch):
    collated = defaultdict(list)

    for x in batch:
        collated["features"].append(x["features"])
        collated["speaker"].append(x["speaker"])
        collated["features_len"].append(x["features_len"])

        if "mels" in x:
            collated["mels"].append(x["mels"])
            collated["mels_len"].append(x["mels_len"])

        if "embeddings" in x:
            collated["embeddings"].append(x["embeddings"])
            collated["embeddings_len"].append(x["embeddings_len"])

        if "transcript" in x:
            collated["transcript"].append(x["transcript"])

    collated["features"] = nn.utils.rnn.pad_sequence(
        collated["features"], batch_first=True
    )
    collated["speaker"] = nn.utils.rnn.pad_sequence(
        collated["speaker"], batch_first=True
    )
    collated["features_len"] = torch.LongTensor(collated["features_len"])

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
