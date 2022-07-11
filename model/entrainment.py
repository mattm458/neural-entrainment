import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from model.components import Attention, Decoder, EmbeddingEncoder, Encoder
from model.util import EmbeddingIterator


class EntrainmentModel(pl.LightningModule):
    modes = tuple(["us", "both"])

    def __init__(
        self,
        lr=0.0001,
        feature_dim=7,
        feature_encoder_out_dim=64,
        feature_encoder_dropout=0.5,
        feature_attention_dim=64,
        decoder_out_dim=64,
        decoder_dropout=0.5,
        training_mode="us",
        training_loss_mode="us",
        evaluation_mode="us",
        evaluation_loss_mode="us",
        teacher_forcing_mode="us",
        embeddings=False,
        embedding_dim=None,
        embedding_encoder_out_dim=None,
        embedding_encoder_dropout=None,
        embedding_attention_dim=None,
        teacher_forcing=0.5,
    ):
        super().__init__()

        self.lr = lr

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

        self.teacher_forcing = teacher_forcing

        self.feature_encoder_out_dim = feature_encoder_out_dim
        self.embedding_attention_dim = embedding_attention_dim

        self.feature_encoder = Encoder(
            in_dim=feature_dim + embedding_encoder_out_dim + 2,
            hidden_dim=feature_encoder_out_dim,
            num_layers=2,
            dropout=feature_encoder_dropout,
        )
        self.feature_attention = Attention(
            history_in_dim=feature_encoder_out_dim,
            latest_in_dim=feature_encoder_out_dim + embedding_encoder_out_dim,
            att_dim=feature_attention_dim,
        )

        self.embeddings = embeddings
        if embeddings:
            self.embedding_encoder = EmbeddingEncoder(
                embedding_dim,
                embedding_encoder_out_dim,
                embedding_encoder_dropout,
                embedding_attention_dim,
            )

        self.decoder = Decoder(
            feature_encoder_out_dim + embedding_encoder_out_dim,
            decoder_out_dim,
            dropout=decoder_dropout,
        )

        self.linear = nn.Sequential(
            nn.Linear(decoder_out_dim, decoder_out_dim),
            nn.ReLU(),
            nn.Linear(decoder_out_dim, feature_dim),
        )

    def forward(
        self,
        i,
        feature_input,
        feature_history,
        feature_mask,
        feature_encoder_hidden,
        decoder_hidden,
        speaker,
        embedding_input=None,
        embedding_len=None,
        pred_embedding_input=None,
        pred_embedding_len=None,
        pred_idxs=None,
    ):
        if self.embeddings:
            if embedding_input is None or embedding_len is None:
                raise Exception("Model requires embedding input but data was missing!")

        batch_size = feature_input.shape[0]

        att_scores = {}

        # If a set of prediction batch indices was not given, then assume we are predicting
        # for the entire batch
        if pred_idxs is None:
            pred_idxs = torch.arange(batch_size)

        output = torch.zeros_like(feature_input)

        # Step 1: Encoder
        # ------------------------------------------------------------------------
        feature_encoder_input = [feature_input]
        feature_encoder_input.append(speaker)

        if self.embeddings:
            # If we are doing word embeddings, we need to encode them first.
            # This step produces a turn-level embedding from the transcript.
            embedding_output, embedding_scores = self.embedding_encoder(
                embedding_input, embedding_len, self.device
            )
            att_scores["embedding"] = embedding_scores

            # If we are looking ahead to word embeddings from the turn we are currently
            # predicting, compute those here
            if len(pred_embedding_input) > 0:
                pred_embedding_output, pred_embedding_scores = self.embedding_encoder(
                    pred_embedding_input, pred_embedding_len, self.device
                )
                att_scores["pred_embedding"] = pred_embedding_scores

            # We treat the turn-level embedding as another feature, so we add
            # it to the feature encoder input.
            feature_encoder_input.append(embedding_output)

        feature_encoder_input = torch.cat(feature_encoder_input, dim=-1)

        feature_encoder_output, feature_encoder_hidden = self.feature_encoder(
            feature_encoder_input, feature_encoder_hidden
        )

        if len(pred_idxs) == 0:
            att_scores["feature"] = torch.zeros((batch_size, i + 1, 1))
            return (
                output,
                att_scores,
                feature_encoder_hidden,
                decoder_hidden,
            )

        # Step 2: Turn-level attention
        # ------------------------------------------------------------------------
        feature_history[:, i] = feature_encoder_output

        # We only want to compute the attention for batches where we are predicting this timestep.
        feature_attention_input = [feature_encoder_output[pred_idxs]]

        # Optionally, concatenate the encoded feature output with the upcoming predicted
        # turn's word embeddings.
        if self.embeddings:
            feature_attention_input.append(pred_embedding_output)

        feature_attention_input = torch.cat(feature_attention_input, dim=-1)

        feature_encoded, feature_scores = self.feature_attention(
            feature_history[pred_idxs, : i + 1],
            feature_attention_input,
            feature_mask[pred_idxs, : i + 1],
        )

        feature_scores_expanded = torch.zeros(
            (batch_size, i + 1, 1), device=self.device
        )
        feature_scores_expanded[pred_idxs] = feature_scores
        att_scores["feature"] = feature_scores_expanded

        # Step 2: Decoder
        # ------------------------------------------------------------------------
        decoder_input = [feature_encoded]

        # Optionally, concatenate the decoder input with the upcoming predicted
        # turn's word embeddings.
        if self.embeddings:
            decoder_input.append(pred_embedding_output)

        decoder_input = torch.cat(decoder_input, dim=-1)

        decoder_output, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, pred_idxs
        )

        output_features = self.linear(decoder_output)

        output = torch.zeros_like(feature_input, dtype=output_features.dtype)
        output[pred_idxs] = output_features

        return output, att_scores, feature_encoder_hidden, decoder_hidden

    def sequence(
        self,
        feature_seq,
        feature_seq_len,
        speaker,
        embedding_seq=None,
        embedding_len=None,
        teacher_forcing=0.5,
    ):
        # Extract some basic information from the sequence of features
        batch_size = feature_seq.shape[0]
        seq_len = feature_seq.shape[1]
        max_len = feature_seq_len.max()

        # Create a placeholder zeroed-out tensor to store the history of encoded turn features
        feature_history = torch.zeros(
            (
                batch_size,
                max_len,
                self.feature_encoder_out_dim,
            ),
            device=self.device,
        )

        # Construct a feature mask allowing us to blank out empty turns
        feature_mask = torch.arange(max_len, device=self.device)
        feature_mask = feature_mask.unsqueeze(0).repeat(len(feature_seq_len), 1)
        feature_mask = feature_mask >= feature_seq_len.unsqueeze(1)
        feature_mask = feature_mask.unsqueeze(2)

        # Create hidden states for the feature encoder and decoder
        feature_encoder_hidden = self.feature_encoder.get_hidden(
            batch_size=batch_size, device=self.device
        )
        decoder_hidden = self.decoder.get_hidden(
            batch_size=batch_size, device=self.device
        )

        # If we have embeddings, set up the iterator that will allow us to retrieve
        # chunks of them
        if self.embeddings:
            embedding_iterator = EmbeddingIterator(
                embedding_seq, embedding_len, batch_size, self.device
            )
            iter(embedding_iterator)

        # Determine whether we are speaking at each turn. A turn can be spoken either by
        # us or our partner. Depending on the settings, the decoder may only run for
        # turns where we are speaking.
        #
        # Our partner always speaks first, so the speaker at timestep 0 is considered our
        # partner.
        partner_id = speaker[:, 0].unsqueeze(1)
        pred_us = (speaker != partner_id).all(-1)[:, 1:]

        feature_input = feature_seq[:, 0]

        outputs = []
        attention_scores = []

        # Main loop: iterate over all turns but the last one, which we are predicting
        # but not using as input
        for i in range(seq_len - 1):
            pred_us_idx = torch.argwhere(pred_us[:, i]).squeeze(-1)

            step_kwargs = {}

            if self.embeddings:
                next_embedding, next_embedding_len = next(embedding_iterator)

                # In our dialogue system, we optionally have access to our output
                # speech in advance. If this is true, then we can use the text
                # we are planning to speak to help decide on acoustic-prosodic features.
                # Retrieve the next embeddings
                y_embedding, y_embedding_len = embedding_iterator.peek()

                step_kwargs["embedding_input"] = next_embedding
                step_kwargs["embedding_len"] = next_embedding_len
                step_kwargs["pred_embedding_input"] = y_embedding[pred_us_idx]
                step_kwargs["pred_embedding_len"] = y_embedding_len[pred_us_idx]

            output_features, scores, feature_encoder_hidden, decoder_hidden = self(
                i=i,
                feature_input=feature_input,
                feature_history=feature_history,
                feature_mask=feature_mask,
                feature_encoder_hidden=feature_encoder_hidden,
                decoder_hidden=decoder_hidden,
                pred_idxs=pred_us_idx,
                speaker=speaker[:, i],
                **step_kwargs,
            )

            if len(output_features) > 0:
                outputs.append(output_features)
            attention_scores.append(scores)

            # Before looping, get the next input features
            feature_input = feature_seq[:, i + 1]

            # Determine which outputs we're going to teacher force
            teacher_force = torch.rand(len(output_features))
            teacher_force = teacher_force < teacher_forcing
            teacher_force_idx = torch.arange(len(output_features))[teacher_force]

            # If we're teacher forcing, copy the decoder outputs where appropriate
            if len(teacher_force_idx) > 0:
                feature_input[teacher_force_idx] = (
                    output_features[teacher_force]
                    .clone()
                    .detach()
                    .type(feature_input.dtype)
                )

        outputs = torch.cat([x.unsqueeze(1) for x in outputs], dim=1)

        return outputs, pred_us, attention_scores

    def validation_step(self, batch, batch_idx):
        outputs, pred_us, attention_scores = self.sequence(
            batch["features"],
            batch["features_len"],
            batch["speaker"],
            batch["embeddings"],
            batch["embeddings_len"],
            teacher_forcing=0.0,
        )

        y = batch["features"][pred_us, 1:]

        loss = F.smooth_l1_loss(outputs[pred_us], y)

        self.log("val_loss", loss, on_epoch=True, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx):
        return self.sequence(
            batch["features"],
            batch["features_len"],
            batch["speaker"],
            batch["embeddings"],
            batch["embeddings_len"],
            teacher_forcing=0.0,
        )

    def training_step(self, batch, batch_idx):
        outputs, pred_us, attention_scores = self.sequence(
            batch["features"],
            batch["features_len"],
            batch["speaker"],
            batch["embeddings"],
            batch["embeddings_len"],
            teacher_forcing=self.teacher_forcing,
        )

        y = batch["features"][pred_us, 1:]

        loss = F.mse_loss(outputs[pred_us], y)

        self.log("train_loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
