""""""

import torch
from app.utils.models import TailModel

async def load_model(model_path: str):
    """Load the Tail Model."""
    # Map to CUDA when available, otherwise load to CPU. This avoids
    # the "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
    # error when running on CPU-only machines.
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(model_path, weights_only=False, map_location=map_location)
    model = TailModel(num_classes=7, dropout=.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model