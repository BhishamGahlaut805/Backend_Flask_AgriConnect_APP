import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

class YieldPredictor(nn.Module):
    def __init__(self, input_size: int, config: Dict):
        super(YieldPredictor, self).__init__()
        self.config = config
        self.input_size = input_size
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None  # Dictionary to store feature importance weights

        # LSTM Encoder with dropout
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            batch_first=True,
            dropout=config['model'].get('dropout', 0.2) if config['model']['num_layers'] > 1 else 0
        )

        # Attention with better initialization
        self.attention = nn.Sequential(
            nn.Linear(config['model']['hidden_size'], config['model']['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_size'], 1, bias=False)
        )

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(config['model']['hidden_size'], config['model']['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_size'], 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all layers"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'attention' in name:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def set_feature_importance(self, importance: Dict[str, float]):
        """
        Set feature importance weights for input features
        Args:
            importance: Dictionary mapping feature names to their importance weights
        """
        self.feature_importance = importance
        self.logger.info(f"Feature importance weights set: {importance}")

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor = None):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            seq_lengths: Optional tensor containing sequence lengths
        Returns:
            Dictionary containing:
            - 'yield': Predicted yield values
            - 'attention': Attention weights
        """
        # Input validation
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be 3D (batch, seq, features), got {x.shape}")

        # Store device for later use
        device = x.device

        # Apply feature importance weighting if set
        if self.feature_importance is not None:
            # Create weight tensor matching input features
            weights = torch.ones(x.shape[-1], device=device, dtype=x.dtype)
            for i in range(x.shape[-1]):
                weights[i] = self.feature_importance.get(f'feature_{i}', 1.0)
            x = x * weights.view(1, 1, -1)

        # Pack sequence if lengths are provided
        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM encoding
        lstm_out, _ = self.encoder(x)

        # Unpack if packed
        if isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence):
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Attention mechanism
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Final prediction
        output = self.output(context).squeeze(-1)

        return {
            'yield': output,
            'attention': attn_weights.squeeze()
        }
        