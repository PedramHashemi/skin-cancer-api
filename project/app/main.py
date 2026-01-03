"""API For Skin Cancer Detection Tool."""

import os
import json
import logging
import torch
import numpy as np
from fastapi import FastAPI, Depends, UploadFile, HTTPException
from torchvision import transforms
from contextlib import asynccontextmanager
# from app.config import get_settings, Settings
from app.utils.tools import load_model
from PIL import Image



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan Context Manager."""
    global pipeline_config, model, pre_transform, device

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    
    # Load pipeline configuration
    with open("./app/configs/pipeline.json", "r") as f:
        pipeline_config = json.load(f)

    # Load the processor with the model
    pre_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=pipeline_config["MEAN"],
            std=pipeline_config["STD"]
        ),
    ])

    # Load Model
    model = await load_model("./app/model/skin_cancer_model_resnet18_20251215_104738.pth")
    model.to(device)
    yield
    # Cleanup code can be added here if necessary

app = FastAPI(
    lifespan=lifespan,
    title="Skin Cancer Detection Tool API",
    description="API for detecting skin cancer using machine learning models.",
    version="0.0.1"
)

@app.get("/")
async def root():
    """Root Endpoint."""
    return {
        "message": "Welcome to the Skin Cancer Detection Tool API",
        "environment": os.environ.get("ENVIRONMENT", "dev")
    }

@app.get("/health")
async def get_health():
    """Health Check Endpoint."""
    return {"status": "ok"}

@app.post("/image/upload/")
async def upload_image(skin_image: UploadFile):
    """Endpoint to upload an image for skin cancer detection."""
    try:
        img = Image.open(skin_image.file)
        img = pre_transform(img).unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).detach().cpu().item()
            print(pred)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file, \n {e}")
    
    return {"class": pred}

