import torch
from torch.utils.data import Dataset


class WindowedFeatureDataset(Dataset):
    def __init__(self, df, tails, tail_length, X_features, target):
        super().__init__()

        self.df = df
        self.tails = tails

        # Generate the X column names from the tail dataframe
        # This will be a list ['x0', 'x1', 'x2', ...] up until the
        # tail length.
        self.tail_x_cols = [f"x{i}" for i in range(tail_length)]

        self.X_features = X_features
        self.target = target

    def __len__(self):
        return len(self.tails)

    def __getitem__(self, idx):
        tail = self.tails.iloc[idx]

        # Get the indexes of all rows in the tail
        X_idxs = tail[self.tail_x_cols].values

        # Get the y index
        y_idx = tail["y"]

        # Return value is a tuple containing:
        #   1. 0 if the target value(s) are from speaker A, 1 if speaker B
        #   2. The input feature(s) of the given tail length
        #   3. The target feature(s)
        return (
            torch.LongTensor([0 if tail.speaker == "A" else 1]),
            torch.FloatTensor(self.df.loc[X_idxs][self.X_features].values.tolist()),
            torch.FloatTensor(self.df.loc[y_idx][self.target].values.tolist()),
        )
