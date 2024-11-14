"""
Configuration file for the xDR (Drug Response Prediction) model.

This module contains various hyperparameters and settings used for training and
configuring the xDR model. It includes settings for training, model architecture,
and loss function calculations.

The configuration adapts based on the availability of CUDA-enabled GPUs, adjusting
batch sizes and epoch counts accordingly.
"""

import torch

# Define a base configuration for all models
BASE_CONFIG = {
    "USE_DATA_TYPES": ["exp", "met", "cop", "mut"],
    "NUM_SAMPLES": None if torch.cuda.is_available() else 5,
    "GENE_FEATURE_DIMENSION": 4,
    "GENE_EDGE_DIMENSION": 7,
    "NUM_GENES": 5181,
    "DROPOUT_RATE": 0.1,
    "HIDDEN_CHANNEL_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "MSE_WEIGHT": 1.1,
    "NUM_ATTENTION_HEADS": 1,
    "NUM_EPOCHS": 70 if torch.cuda.is_available() else 2,
    "NUM_GNN_LAYERS": 3,
    "USE_TRANSFORMER": False,
    "OUTPUT_CHANNEL_SIZE": 1,
    "MODEL_NAME": None,  # Added MODEL_NAME
}


# Function to return model-specific configurations based on the model type
def get_model_config(model_type, base_config=BASE_CONFIG):
    model_configs = {
        "GAT": {
            "BATCH_SIZE": 100 if torch.cuda.is_available() else 5,
            "IMPORTANCE_DECAY": 0.9,
            "IMPORTANCE_REGULARIZATION_WEIGHT": 0.01,
            "IMPORTANCE_THRESHOLD": 1e-05,
            "MODEL_NAME": "GAT",  # Added MODEL_NAME
        },
        "GCN": {
            "BATCH_SIZE": 70 if torch.cuda.is_available() else 5,
            "IMPORTANCE_DECAY": 0.8,
            "IMPORTANCE_REGULARIZATION_WEIGHT": 0.01,
            "IMPORTANCE_THRESHOLD": 0.001,
            "layer_type": "gcn",
            "MODEL_NAME": "GCN",  # Added MODEL_NAME
        },
        "MPNN": {
            "BATCH_SIZE": 100 if torch.cuda.is_available() else 5,
            "IMPORTANCE_DECAY": 0.8,
            "IMPORTANCE_REGULARIZATION_WEIGHT": 0.01,
            "IMPORTANCE_THRESHOLD": 0.001,
            "layer_type": "mpnn",
            "MODEL_NAME": "MPNN",  # Added MODEL_NAME
        },
        "GT": {
            "BATCH_SIZE": 70 if torch.cuda.is_available() else 5,
            "IMPORTANCE_DECAY": 0.8,
            "IMPORTANCE_REGULARIZATION_WEIGHT": 0.02,
            "IMPORTANCE_THRESHOLD": 0.001,
            "USE_TRANSFORMER": True,
            "MODEL_NAME": "GT",  # Added MODEL_NAME
        },
        "GINE": {
            "BATCH_SIZE": 100 if torch.cuda.is_available() else 5,
            "IMPORTANCE_DECAY": 0.9,
            "IMPORTANCE_REGULARIZATION_WEIGHT": 0.02,
            "IMPORTANCE_THRESHOLD": 1e-05,
            "MSE_WEIGHT": 1.0,
            "MODEL_NAME": "GINE",  # Added MODEL_NAME
        },
    }
    config = base_config.copy()
    config.update(model_configs.get(model_type, {}))
    print(config)
    return config


# Example usage
# model_config = get_model_config('GAT', BASE_CONFIG)
