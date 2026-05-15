import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import logging

from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler

from .lstm_model import YieldPredictor
from .model_io import save_lstm_model


class PaddedSequenceDataset(Dataset):

    def __init__(
        self,
        sequences: List[np.ndarray],
        targets: List[float],
        scaler_y: MinMaxScaler = None
    ):

        self.sequences = sequences
        self.targets = targets
        self.scaler_y = scaler_y

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        target = self.targets[idx]

        if self.scaler_y:

            target = self.scaler_y.transform(
                [[target]]
            )[0][0]

        return {
            "sequence": self.sequences[idx].astype(np.float32),
            "target": float(target),
            "length": len(self.sequences[idx])
        }


def collate_fn(batch):

    batch.sort(
        key=lambda x: x["length"],
        reverse=True
    )

    sequences = [
        torch.from_numpy(item["sequence"])
        for item in batch
    ]

    lengths = torch.tensor(
        [item["length"] for item in batch],
        dtype=torch.long
    )

    targets = torch.tensor(
        [item["target"] for item in batch],
        dtype=torch.float32
    )

    padded_seqs = nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True
    )

    return {
        "sequences": padded_seqs,
        "targets": targets,
        "lengths": lengths
    }


class LSTMTrainer:

    def __init__(self, config: Dict):

        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.logger = logging.getLogger(__name__)

        self.scaler_y = MinMaxScaler(
            feature_range=(0, 1)
        )

    def train(
        self,
        sequences: List[np.ndarray],
        targets: List[float],
        farm_id: str,
        crop_name: str
    ):

        try:

            # Scale targets
            targets_scaled = self.scaler_y.fit_transform(
                np.array(targets).reshape(-1, 1)
            ).flatten()

            # Dataset
            dataset = PaddedSequenceDataset(
                sequences,
                targets_scaled,
                self.scaler_y
            )

            # DataLoader
            loader = DataLoader(
                dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True,
                collate_fn=collate_fn
            )

            # Input size
            input_size = sequences[0].shape[1]

            # Initialize model
            model = YieldPredictor(
                input_size,
                self.config
            ).to(self.device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config["training"]["lr"]
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=5
            )

            criterion = nn.MSELoss()

            best_loss = float("inf")

            early_stop_counter = 0

            # Training loop
            for epoch in range(
                self.config["training"]["epochs"]
            ):

                model.train()

                epoch_loss = 0.0

                for batch in loader:

                    inputs = batch["sequences"].to(self.device)

                    targets = batch["targets"].to(self.device)

                    lengths = batch["lengths"].to(self.device)

                    optimizer.zero_grad()

                    outputs = model(
                        inputs,
                        lengths
                    )

                    loss = criterion(
                        outputs["yield"],
                        targets
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0
                    )

                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)

                scheduler.step(avg_loss)

                self.logger.info(
                    f"Epoch {epoch+1} | "
                    f"Loss: {avg_loss:.6f}"
                )

                # Save best model
                if avg_loss < (
                    best_loss -
                    self.config["training"]["min_delta"]
                ):

                    best_loss = avg_loss

                    early_stop_counter = 0

                    self._save_model(
                        model,
                        farm_id,
                        crop_name
                    )

                else:

                    early_stop_counter += 1

                    if early_stop_counter >= self.config["training"]["patience"]:

                        self.logger.info(
                            "Early stopping triggered"
                        )

                        break

            return "MODEL_SAVED_TO_MONGODB"

        except Exception as e:

            self.logger.error(
                f"Training failed: {str(e)}",
                exc_info=True
            )

            raise

    def _save_model(
        self,
        model,
        farm_id,
        crop_name
    ):

        try:

            save_lstm_model(
                model=model,
                scalers={
                    "yield_scaler": self.scaler_y
                },
                farm_id=farm_id,
                crop_name=crop_name,
                config=self.config
            )

            self.logger.info(
                f"Model saved successfully "
                f"to MongoDB GridFS | "
                f"Farm={farm_id} | Crop={crop_name}"
            )

        except Exception as e:

            self.logger.error(
                f"Failed to save model: {str(e)}",
                exc_info=True
            )

            raise
