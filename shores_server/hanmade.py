

from .tasks import segment, testLoadAndSaveMasks

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pprint import pprint

#image = cv2.imread('../data/q2008.tif')
image = 'q2008.tif'
masks = 'q2008.picle'

# masks=segment(image)

testLoadAndSaveMasks(image, masks, 'recognized.tiff')

pprint(masks)
