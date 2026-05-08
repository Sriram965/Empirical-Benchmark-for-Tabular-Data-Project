import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping




# MLP Model
class MLPModule(nn.Module):
   
    def __init__(self, input_dim=10, output_dim=2,
                 n_layers=3, hidden_dim=128, dropout=0.1):
        
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ResNet Model

class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetModule(nn.Module):
    
    def __init__(self, input_dim=10, output_dim=2,
                 n_blocks=3, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.output(x)


# FT-Transformer Model
class FeatureTokenizer(nn.Module):
    
    def __init__(self, input_dim, token_dim):
        super().__init__()
        self.weight    = nn.Parameter(torch.empty(input_dim, token_dim))
        self.bias      = nn.Parameter(torch.empty(input_dim, token_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [batch, n_features] → tokens: [batch, n_features, token_dim]
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls    = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformerModule(nn.Module):
    
    def __init__(self, input_dim=10, output_dim=2,
                 n_blocks=3, token_dim=64, n_heads=8, dropout=0.1):
        super().__init__()

        # token_dim must be divisible by n_heads for multi-head attention
        token_dim = (token_dim // n_heads) * n_heads

        self.tokenizer = FeatureTokenizer(input_dim, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_blocks)
        self.output = nn.Linear(token_dim, output_dim)

    def forward(self, x):
        tokens  = self.tokenizer(x)
        encoded = self.transformer(tokens)
        return self.output(encoded[:, 0, :])  # use CLS token only


# Skorch Factory Functions 

def get_mlp(trial, task_type, input_dim, output_dim):
    
    params = {
        'n_layers':     trial.suggest_int('mlp_n_layers', 1, 5),
        'hidden_dim':   trial.suggest_int('mlp_hidden_dim', 64, 512),
        'dropout':      trial.suggest_float('mlp_dropout', 0.0, 0.5),
        'lr':           trial.suggest_float('mlp_lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('mlp_weight_decay', 1e-6, 1e-2, log=True),
    }

    net_class = NeuralNetClassifier if task_type == 'classification' else NeuralNetRegressor
    return net_class(
        module=MLPModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__n_layers=params['n_layers'],
        module__hidden_dim=params['hidden_dim'],
        module__dropout=params['dropout'],
        lr=params['lr'],
        optimizer__weight_decay=params['weight_decay'],
        max_epochs=200,
        batch_size=256,
        callbacks=[EarlyStopping(patience=20)],
        verbose=0   
    )


def get_resnet(trial, task_type, input_dim, output_dim):

    params = {
        'n_blocks':     trial.suggest_int('resnet_n_blocks', 1, 6),
        'hidden_dim':   trial.suggest_int('resnet_hidden_dim', 64, 512),
        'dropout':      trial.suggest_float('resnet_dropout', 0.0, 0.5),
        'lr':           trial.suggest_float('resnet_lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('resnet_weight_decay', 1e-6, 1e-2, log=True),
    }

    net_class = NeuralNetClassifier if task_type == 'classification' else NeuralNetRegressor
    return net_class(
        module=ResNetModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__n_blocks=params['n_blocks'],
        module__hidden_dim=params['hidden_dim'],
        module__dropout=params['dropout'],
        lr=params['lr'],
        optimizer__weight_decay=params['weight_decay'],
        max_epochs=200,
        batch_size=256,
        callbacks=[EarlyStopping(patience=20)],
        verbose=0
    )


def get_ft_transformer(trial, task_type, input_dim, output_dim):
    
    n_heads   = trial.suggest_int('ftt_n_heads', 2, 8)
    token_dim = trial.suggest_int('ftt_token_dim', 32, 256)
    token_dim = (token_dim // n_heads) * n_heads 

    params = {
        'n_blocks':     trial.suggest_int('ftt_n_blocks', 1, 4),
        'token_dim':    token_dim,
        'n_heads':      n_heads,
        'dropout':      trial.suggest_float('ftt_dropout', 0.0, 0.3),
        'lr':           trial.suggest_float('ftt_lr', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('ftt_weight_decay', 1e-6, 1e-2, log=True),
    }

    net_class = NeuralNetClassifier if task_type == 'classification' else NeuralNetRegressor
    return net_class(
        module=FTTransformerModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__n_blocks=params['n_blocks'],
        module__token_dim=params['token_dim'],
        module__n_heads=params['n_heads'],
        module__dropout=params['dropout'],
        lr=params['lr'],
        optimizer__weight_decay=params['weight_decay'],
        max_epochs=200,
        batch_size=256,
        callbacks=[EarlyStopping(patience=20)],
        verbose=0
    )