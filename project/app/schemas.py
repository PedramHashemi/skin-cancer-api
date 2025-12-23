"""Schemas Module.

Acceptable data types for API requests and responses are defined here.
"""
import pydantic import BaseModel
import numpy as np

class ImageClass(BaseModel):
    img_format: str