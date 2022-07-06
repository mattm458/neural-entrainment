import pytorch_lightning as pl
import torch
from torch import functional as F
from torch import nn

from model.components import Attention, Decoder, Encoder, MelEncoder


class EntrainmentModel(pl.LightningModule):
    modes = tuple(["us", "both"])

    def __init__(
        self,
        lr=0.0001,
        rnn_in_dim=3,
        rnn_out_dim=64,
        out_dim=1,
        teacher_forcing=0.5,
        att_dim=128,
        teacher_forcing_schedule=False,
        teacher_forcing_end=0.5,
        teacher_forcing_schedule_start_epochs=10,
        teacher_forcing_schedule_transition_epochs=10,
        dropout=0.5,
        training_mode="us",
        training_loss_mode="us",
        evaluation_mode="us",
        evaluation_loss_mode="us",
        teacher_forcing_mode="us",
        encoder_speaker=True,
        attention_speaker=True,
        decoder_speaker=True,
        mels=False,
        embeddings=False,
        vectors=None,
    ):
        super().__init__()

        self.dim_multiplier = 2 if mels else 1

        # self.save_hyperparameters()

        if training_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid training mode {training_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.training_mode = training_mode

        if training_loss_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid training loss mode {training_loss_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.training_loss_mode = training_loss_mode

        if evaluation_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid evaluation mode {evaluation_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.evaluation_mode = evaluation_mode

        if evaluation_loss_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid evaluation loss mode {evaluation_loss_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.evaluation_loss_mode = evaluation_loss_mode

        if teacher_forcing_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid teacher forcing mode {teacher_forcing_mode}, legal values are {EntrainmentModel.modes}"
            )
        self.teacher_forcing_mode = teacher_forcing_mode

        self.encoder_speaker = encoder_speaker
        self.attention_speaker = attention_speaker
        self.decoder_speaker = decoder_speaker

        self.rnn_in_dim = rnn_in_dim
        self.lr = lr
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_start = teacher_forcing
        self.teacher_forcing_schedule = teacher_forcing_schedule
        self.teacher_forcing_end = teacher_forcing_end
        self.teacher_forcing_schedule_start_epochs = (
            teacher_forcing_schedule_start_epochs
        )
        self.teacher_forcing_schedule_transition_epochs = (
            teacher_forcing_schedule_transition_epochs
        )
        self.rnn_out_dim = rnn_out_dim

        self.encoder = Encoder(
            rnn_in_dim + (2 if encoder_speaker else 0),
            rnn_out_dim,
            dropout=dropout,
        )
        self.attention = Attention(
            (rnn_out_dim * 2),
            (rnn_out_dim * 2) + (2 if attention_speaker else 0),
            att_dim,
        )
        self.decoder = Decoder(
            (rnn_out_dim * 2) + (2 if decoder_speaker else 0),
            rnn_out_dim,
            dropout=dropout,
        )
        self.linear = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim),
            nn.ReLU(),
            nn.Linear(rnn_out_dim, out_dim),
        )

        self.mels = mels
        if mels:
            self.mel_encoder = MelEncoder(out_dim=rnn_out_dim)

        self.embeddings = embeddings
        if embeddings:
            self.vectors = vectors

            self.embedding_encoder = nn.LSTM(
                300,
                rnn_out_dim // 2,
                bidirectional=True,
                num_layers=2,
                batch_first=True,
            )
            self.embedding_att = Attention(
                rnn_out_dim,
                (rnn_out_dim * 2) + (2 if decoder_speaker else 0),
                rnn_out_dim,
            )

    def training_epoch_end(self, outputs):
        self.log("teacher_forcing", self.teacher_forcing, on_epoch=True)

        if self.teacher_forcing_schedule:
            teacher_forcing_epoch = (
                self.current_epoch - self.teacher_forcing_schedule_start_epochs
            )

            if (
                teacher_forcing_epoch >= 0
                and teacher_forcing_epoch
                < self.teacher_forcing_schedule_transition_epochs
            ):
                diff = self.teacher_forcing_start - self.teacher_forcing_end
                diff /= self.teacher_forcing_schedule_transition_epochs
                self.teacher_forcing -= diff

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        predictions, predictions_mask, scores = self(
            batch["X_features"],
            batch["X_speaker"],
            batch["X_features_len"],
            batch["y_features"],
            batch["y_speaker"],
            teacher_forcing=self.teacher_forcing,
            mode=self.training_mode,
            loss_mode=self.training_loss_mode,
            embeddings=batch["embeddings"],
            embeddings_len=batch["embeddings_len"],
        )

        loss = F.mse_loss(
            predictions[predictions_mask], batch["y_features"][predictions_mask]
        )

        self.log("train_loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        predictions, predictions_mask, scores = self(
            batch["X_features"],
            batch["X_speaker"],
            batch["X_features_len"],
            batch["y_features"],
            batch["y_speaker"],
            teacher_forcing=0.0,
            mode=self.evaluation_mode,
            loss_mode=self.evaluation_loss_mode,
            embeddings=batch["embeddings"],
            embeddings_len=batch["embeddings_len"],
        )

        loss = F.smooth_l1_loss(
            predictions[predictions_mask], batch["y_features"][predictions_mask]
        )

        self.log("val_loss", loss, on_epoch=True, on_step=True)

        return loss

    def forward(
        self,
        X_features,
        X_speaker,
        X_len,
        y_features,
        y_speaker,
        mels=None,
        mels_len=None,
        teacher_forcing=0.0,
        mode="us",
        loss_mode="us",
        embeddings=None,
        embeddings_len=None,
    ):

        if mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid mode {mode}, legal values are {EntrainmentModel.modes}"
            )

        if loss_mode not in EntrainmentModel.modes:
            raise Exception(
                f"Invalid loss mode {loss_mode}, legal values are {EntrainmentModel.modes}"
            )

        batch_size = X_features.shape[0]
        max_len = X_len.max()

        mask = torch.arange(max_len, device=self.device).unsqueeze(0).repeat(
            len(X_len), 1
        ) >= X_len.unsqueeze(1)
        mask = mask.unsqueeze(2)

        encoder_hidden = self.encoder.get_hidden(batch_size, self.device)
        decoder_hidden = self.decoder.get_hidden(batch_size, self.device)

        history = torch.zeros(
            batch_size, max_len, self.rnn_out_dim * 2, device=self.device
        )

        # From the turns we're about to predict, identify those which are uttered by our partner.
        # Our partner is always the first speaker in the conversation.
        them = X_speaker[:, 0]

        # From the turns we're about to predict, identify those which we speak.
        us = (y_speaker != them.unsqueeze(1)).all(2)

        if mode == "us":
            us_mask = us
        elif mode == "both":
            us_mask = ~mask.squeeze(-1)

        if self.teacher_forcing_mode == "us":
            teacher_forcing_mask = us
        elif self.teacher_forcing_mode == "both":
            teacher_forcing_mask = ~mask.squeeze(-1)

        # next_mel_start = 0
        # next_mel_len = mels_len[:, 0]
        # max_mel_len = next_mel_len.max()
        # next_mel = mels[:, next_mel_start:next_mel_start+max_mel_len]

        predictions = []
        predictions_mask = []
        scores = []

        next_embedding_start = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )

        for i in range(max_len):
            next_input = X_features[:, i]

            next_embedding = []
            next_embedding_len = embeddings_len[:, i]
            max_embedding_len = next_embedding_len.max()

            for batch_embeddings, start, length in zip(
                embeddings, next_embedding_start, next_embedding_len
            ):
                next_embedding.append(batch_embeddings[start : start + length])
            next_embedding = nn.utils.rnn.pad_sequence(
                next_embedding, batch_first=True, padding_value=1
            )
            next_embedding_start += next_embedding_len

            # mel_encoded = self.mel_encoder(next_mel, next_mel_len)

            # Step 1: Encoder
            # ------------------------------------------------------------------------
            # Encode everything regardless of who we are predicting right now
            # encoder_input = torch.cat(
            #     [next_input, mel_encoded, X_speaker[:, i]], dim=-1
            # )
            encoder_input = torch.cat([next_input, X_speaker[:, i]], dim=-1)
            encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)

            embedding_encoder_input = nn.utils.rnn.pack_padded_sequence(
                next_embedding,
                next_embedding_len.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            embedding_encoder_output, (embedding_h, _) = self.embedding_encoder(
                embedding_encoder_input
            )
            embedding_encoder_output, _ = nn.utils.rnn.pad_packed_sequence(
                embedding_encoder_output, batch_first=True
            )

            # number of layers, directions, batch_size, hidden_dim
            embedding_h = embedding_h.reshape(2, 2, batch_size, -1)
            embedding_h = embedding_h[-1].reshape(batch_size, -1)

            # Attention embedding
            embedding_mask = torch.arange(
                max_embedding_len, device=self.device
            ).unsqueeze(0).repeat(
                len(next_embedding_len), 1
            ) >= next_embedding_len.unsqueeze(
                1
            )
            embedding_mask = embedding_mask.unsqueeze(2)
            embedding_att_input = [embedding_h, encoder_output]
            if self.attention_speaker:
                embedding_att_input.append(y_speaker[:, i])
            embedding_att_input = torch.cat(embedding_att_input, dim=1)
            embedding_att_output, embedding_score = self.embedding_att(
                embedding_encoder_output,
                embedding_att_input,
                embedding_mask,
            )

            history[:, i] = torch.cat([encoder_output, embedding_att_output], dim=1)

            # Step 2: Preparation for decoding
            # ------------------------------------------------------------------------
            # Create a mask we can use later to isolate outputs that we are actually
            # predicting at this timestep.
            prediction_mask = us_mask[:, i]

            # Transform the mask into a list of indices.
            prediction_idxs = torch.argwhere(prediction_mask).squeeze(1)

            # If we aren't predicting anything this timestep, skip the rest of it
            if len(prediction_idxs) == 0:
                # Since we aren't predicting anything this timestep, fill our prediections
                # with a zero vector
                predictions.append(
                    torch.zeros_like(next_input, device=self.device).unsqueeze(1)
                )

                continue

            # Step 3: Attention
            # ------------------------------------------------------------------------
            # Compute attention over any of the inputs we are predicting
            att_input = [history[:, i]]
            if self.attention_speaker:
                att_input.append(y_speaker[:, i])

            att_input = torch.cat(att_input, dim=1)

            att_output, score = self.attention(
                history[prediction_idxs, : i + 1],
                att_input[prediction_idxs],
                mask[prediction_idxs, : i + 1],
            )
            score_expanded = torch.zeros((batch_size, i + 1), device=self.device)
            score_expanded[prediction_idxs] = score.squeeze(-1)
            scores.append(score_expanded)

            # embedding_history[:, i] = embedding_att_output
            # embedding_score_expanded = torch.zeros((batch_size, embedding_score.shape[1]), device=self.device)
            # embedding_score_expanded[prediction_idxs] = embedding_score.squeeze(-1)
            # scores.append(score_expanded)

            # Step 3: Decoding
            # ------------------------------------------------------------------------
            # Decode the attention output into predicted acoustic/prosodic values
            decoder_input = [att_output]
            if self.decoder_speaker:
                decoder_input.append(y_speaker[prediction_idxs, i])

            decoder_input = torch.cat(decoder_input, dim=1)
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, prediction_idxs
            )

            linear_out = self.linear(decoder_output)

            # Cleanup - project the predicted values into a larger space of the same size
            # as the input. All values we did not predict will be zeros.
            prediction = torch.zeros_like(next_input)
            prediction[prediction_idxs] = linear_out.type(prediction.dtype)
            predictions.append(prediction.unsqueeze(1))

            # Step 3: Teacher forcing
            # ------------------------------------------------------------------------
            # Determine what values we want to feed back into the encoder.
            # It can either be ground truth values or our own predictions.
            # However, since we do not predict our partner's values, we will always
            # feed their ground-truth values back into the input. Teacher forcing
            # only applies to our own outputs.

            # Initially, set next_input to all ground truth values.
            next_input = y_features[:, i].clone()

            teacher_forcing_eligible = teacher_forcing_mask[prediction_idxs, i]
            teacher_forcing_eligible_idx = torch.argwhere(
                teacher_forcing_eligible
            ).squeeze(1)

            # Decide which of the predicted values should be copied.
            # autoregress = (
            #     torch.rand(prediction_idxs.shape, device=self.device) >= teacher_forcing
            # )
            autoregress = (
                torch.rand(teacher_forcing_eligible_idx.shape, device=self.device)
                >= teacher_forcing
            )

            # Using the teacher forcing tensor, determine the indices of values which should be
            # copied from our predictions.
            # autoregress_idx = prediction_idxs[autoregress]
            autoregress_idx = teacher_forcing_eligible_idx[autoregress]

            # Copy any selected values from our prediction to the next input
            # next_input[autoregress_idx] = (
            #     linear_out[autoregress].detach().clone().type(next_input.dtype)
            # )
            next_input[autoregress_idx] = (
                linear_out[autoregress_idx].detach().clone().type(next_input.dtype)
            )

            # next_mel_start = max_mel_len
            # next_mel_len = mels_len[:, i]
            # max_mel_len = next_mel_start + next_mel_len.max()
            # next_mel = mels[:, next_mel_start:max_mel_len]

        predictions = torch.cat(predictions, dim=1)

        if loss_mode == "us":
            predictions_mask = us
        elif loss_mode == "both":
            predictions_mask = ~mask.squeeze(-1)

        return predictions, predictions_mask, scores
