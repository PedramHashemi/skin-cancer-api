""""""
import torch
import asyncio
from models import TailModel

async def load_model(model_path: str):
    """Load the Tail Model."""

    model = TailModel()
    model = model.load_state_dict(torch.load("/models/tail_model.pth"))
    return model