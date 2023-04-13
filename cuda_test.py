import sys
import torch
import pandas as pd
import numpy as np
import scipy
import PIL
import sklearn
import matplotlib
import h5py
import pickle
import cv2
import warnings

warnings.simplefilter('ignore')

import torchvision

sys.path.append("utils/openslide/openslide")
import openslide

print("# numpy (1.23.5)")
print(f"- version: {np.__version__}")

print("# pandas (1.5.3)")
print(f"- version: {pd.__version__}")

print("# scipy (1.10.1)")
print(f"- version: {scipy.__version__}")

print("# sklearn (1.2.2)")
print(f"- version: {sklearn.__version__}")

print("# matplotlib (3.7.1)")
print(f"- version: {matplotlib.__version__}")

print("# h5py (3.7.0)")
print(f"- version: {h5py.__version__}")

print("# pickle (???)")
print(f"- package class: {pickle}")

print("# torch (depend on env)")
print(f"- version: {torch.__version__}")
print(f"- cuda available: {torch.cuda.is_available()}")
print(f"- cuda device: {torch.cuda.current_device()}")

print("# torchvision (depend on env)")
print(f"- version: {torchvision.__version__}")

print("# PIL (9.4.0)")
print(f"- version: {PIL.__version__}")

print("# opencv (4.7.0)")
print(f"- version: {cv2.__version__}")

print("# openslide (1.2.0)")
print(f"- version: {openslide.__version__}")