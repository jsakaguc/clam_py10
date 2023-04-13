import sys
import torch
import pandas as pd

print("# pandas test")
print(pd.__version__)

print("# torch test")
print(torch.cuda.is_available())
print(torch.cuda.current_device())

sys.path.append("utils/openslide/openslide")
import openslide