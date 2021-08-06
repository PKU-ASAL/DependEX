import os

import numpy as np
import torch
from .Transformer import Transformer

def setup(args):
	# Transformer
	model = Transformer(args)
	return model
