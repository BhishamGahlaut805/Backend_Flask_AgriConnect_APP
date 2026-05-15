import torch    #type: ignore
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional, List
import os
import pandas as pd
from .lstm_model import YieldPredictor
# from torch.serialization import add_safe_globals #type:ignore
from sklearn.preprocessing import MinMaxScaler

# Add MinMaxScaler to safe globals before loading the model
# add_safe_globals([MinMaxScaler])
class LSTMPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

    def load_model(self, farm_id: str, crop_name: str) -> Optional[Dict]:
        """Load trained model with all components"""
        try:
            model_dir = os.path.join(self.config['model_dir'], farm_id, "crops", crop_name)
            checkpoint_path = os.path.join(model_dir, "model_best.pt")

            if not os.path.exists(checkpoint_path):
                self.logger.error(f"No model found at {checkpoint_path}")
                return None

            # Load with weights_only=False since we've added MinMaxScaler to safe globals
            checkpoint = torch.load(checkpoint_path,
                                 map_location=self.device,
                                 weights_only=False)

            # Reconstruct model
            model = YieldPredictor(
                input_size=checkpoint['input_size'],
                config=checkpoint['config']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state'])
            model.set_feature_importance(checkpoint.get('feature_importance', {}))

            return {
                'model': model,
                'scaler_y': checkpoint['scaler_y'],
                'config': checkpoint['config'],
                'timestamp': checkpoint['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            return None

    def predict(self, model_data: Dict, df: pd.DataFrame):
        try:
            # Prepare input sequence
            features = [col for col in df.columns if col in model_data['config']['data']['feature_columns']]
            current_seq = df[features].values.astype(np.float32)

            # Convert to tensor
            seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
            seq_length = torch.tensor([current_seq.shape[0]]).to(self.device)

            # Predict
            model_data['model'].eval()
            with torch.no_grad():
                outputs = model_data['model'](seq_tensor, seq_length)

            # Inverse transform the prediction
            predicted_yield = model_data['scaler_y'].inverse_transform(
                outputs['yield'].cpu().numpy().reshape(-1, 1)
            )[0][0]

            return {
                'predicted_yield': float(predicted_yield),
                'attention_weights': outputs['attention'].cpu().numpy().tolist()
            }

        except Exception as e:
            return {'error': str(e)}

    def what_if(self, model_data: Dict, df: pd.DataFrame, scenarios: List[Dict]) -> Dict:
        """Run what-if scenario analysis"""
        try:
            baseline = self.predict(model_data, df)
            if 'error' in baseline:
                return baseline

            results = {'baseline': baseline}

            for scenario in scenarios:
                modified_df = df.copy()
                for feature, change in scenario['changes'].items():
                    if feature in modified_df.columns:
                        if change.get('type', 'percentage') == 'percentage':
                            modified_df[feature] *= (1 + change['value']/100)
                        else:
                            modified_df[feature] += change['value']

                scenario_result = self.predict(model_data, modified_df)
                if 'error' not in scenario_result:
                    results[scenario['name']] = {
                        'prediction': scenario_result['predicted_yield'],
                        'difference': scenario_result['predicted_yield'] - baseline['predicted_yield'],
                        'attention_changes': {
                            date: {
                                'baseline': baseline['attention_weights'].get(date, 0),
                                'scenario': scenario_result['attention_weights'].get(date, 0),
                                'change': (scenario_result['attention_weights'].get(date, 0) -
                                          baseline['attention_weights'].get(date, 0))
                            }
                            for date in baseline['attention_weights']
                        }
                    }

            return results

        except Exception as e:
            self.logger.error(f"What-if analysis failed: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _calculate_feature_contributions(self, model, seq_tensor, seq_length, feature_names):
        """Calculate feature contributions using integrated gradients"""
        # Baseline is zero input
        baseline = torch.zeros_like(seq_tensor).to(self.device)

        # Number of steps for approximation
        steps = 50
        total_grads = 0

        # Disable dropout
        model.eval()

        for alpha in torch.linspace(0, 1, steps):
            # Interpolated input
            input = baseline + alpha * (seq_tensor - baseline)
            input.requires_grad = True

            # Forward pass
            output = model(input, seq_length)['yield']

            # Backward pass
            output.backward()

            # Accumulate gradients
            total_grads += input.grad.detach()

            # Zero gradients
            model.zero_grad()
            input.grad = None

        # Integrated gradients
        ig = (seq_tensor - baseline) * total_grads / steps

        # Average over sequence length
        feature_contrib = ig.mean(dim=1).squeeze().cpu().numpy()

        return {
            name: float(contrib)
            for name, contrib in zip(feature_names, feature_contrib)
        }
