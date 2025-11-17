

from .tasks import segment, testLoadAndSaveMasks

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pprint import pprint

# import gdalcopyproj as gcp

#image = cv2.imread('../data/q2008.tif')
image = 'q2008.tif'
masks = 'q2008.picle'


import os

PY='/home/eugeneai/.pyenv/versions/shores_server/bin/python'

for iname, name, img in testLoadAndSaveMasks(image, masks, 'recognized.tiff', gen=True):
    os.system("{} gdalcopyproj.py {} {}".format(PY, iname, name))

pprint(masks)
