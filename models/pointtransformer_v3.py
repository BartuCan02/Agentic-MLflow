"""
models/pointtransformer_v3.py
--------------------------------
Wrapper to use the Point-Transformer-Pytorch library.
This file now exports the basic building block, PointTransformerLayer.
"""
from point_transformer_pytorch import PointTransformerLayer

# The agents will use this layer to build full models.
PointTransformerV3Layer = PointTransformerLayer


