import torch
import torchaudio
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.prenet = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, encoder_input, hidden):
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = self.prenet(encoder_input)

        hidden_out = []

        for h, rnn in zip(hidden, self.rnn):
            (h_out, c_out) = rnn(x, h)
            x = self.dropout(h_out)
            hidden_out.append((x, c_out))

        return x, hidden_out


class Attention(nn.Module):
    def __init__(self, history_in_dim, latest_in_dim, att_dim):
        super().__init__()

        self.att_w1 = nn.Linear(history_in_dim, att_dim, bias=False)
        self.att_w2 = nn.Linear(latest_in_dim, att_dim, bias=False)
        self.att_v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, history, latest, mask):
        w1 = self.att_w1(history)
        w2 = self.att_w2(latest).unsqueeze(1)
        score = self.att_v(torch.tanh(w1 + w2))

        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)
        score_out = score.detach()
        score = score.squeeze(-1)

        score = score.unsqueeze(1)
        att_applied = torch.bmm(score, history)
        att_applied = att_applied.squeeze(1)

        return att_applied, score_out


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.ModuleList([nn.LSTMCell(in_dim, hidden_dim)])
        for i in range(num_layers - 1):
            self.rnn.append(nn.LSTMCell(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, decoder_input, hidden, at_idx=None):
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        hidden_out = []

        x = decoder_input

        for (h_in, c_in), rnn in zip(hidden, self.rnn):
            hidden = (
                (h_in[at_idx], c_in[at_idx]) if at_idx is not None else (h_in, c_in)
            )

            (h_out, c_out) = rnn(x, hidden)
            x = self.dropout(h_out)

            if at_idx is not None:
                h_in[at_idx] = h_out.type(h_in.dtype)
                h_out = h_in
                c_in[at_idx] = c_out.type(c_in.dtype)
                c_out = c_in

            hidden_out.append((h_out, c_out))

        return x, hidden_out


class MelEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.delta = torchaudio.transforms.ComputeDeltas()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, (5, 3), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(128, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            # nn.LeakyReLU(),
        )

        self.pre_rnn = nn.Sequential(nn.Linear(10 * 256, 768), nn.LeakyReLU())
        self.rnn = nn.GRU(768, 128, batch_first=True, bidirectional=True)

        self.frame_weight = nn.Linear(256, 256)
        self.context_weight = nn.Linear(256, 1)

        self.linear = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, out_dim)
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        device = "cuda" if mel_spectrogram.get_device() >= 0 else "cpu"

        mel_spectrogram = mel_spectrogram.swapaxes(1, 2)

        if mel_spectrogram.shape[2] % 2 == 1:
            mel_spectrogram = torch.cat(
                [
                    mel_spectrogram,
                    torch.zeros(
                        (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1),
                        device=device,
                    ),
                ],
                2,
            )

        d1 = self.delta(mel_spectrogram)
        d2 = self.delta(d1)

        x = torch.cat(
            [mel_spectrogram.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)], dim=1
        ).swapaxes(2, 3)

        output = self.conv(x)
        output = output.permute(0, 2, 3, 1).reshape(
            mel_spectrogram.shape[0], mel_spectrogram.shape[2], 256 * 10
        )

        output = self.pre_rnn(output)

        output, _ = self.rnn(output)

        att_output = self.frame_weight(output)
        att_output = self.context_weight(att_output)

        mask = torch.arange(att_output.shape[1], device=device).unsqueeze(0).repeat(
            len(mel_spectrogram_len), 1
        ) >= mel_spectrogram_len.unsqueeze(1)
        mask = mask.unsqueeze(2)

        att_output = att_output.masked_fill(mask, 1e-4)
        att_output = torch.softmax(att_output, 1)
        output = (output * att_output).sum(1)

        return self.linear(output)
