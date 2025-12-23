""""""

import torch
from app.utils.models import TailModel

async def load_model(model_path: str):
    """Load the Tail Model."""

    checkpoint = torch.load(model_path, weights_only=False)
    model = TailModel(num_classes=7, dropout=.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model