#########################################################################################################
#
#   This file's purpose is to test the previously trained model against the Sartorius Cell dataset.
#
#########################################################################################################
#
#   The code in this file is based on :-
#   https://www.kaggle.com/slawekbiel/positive-score-with-detectron-3-3-inference
#
#########################################################################################################

# Import tools.
# Ignore unused warning, we are using function calls beneath imported directories.
import detectron2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastcore.all import *
             
# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# This gives us the resulting mask of instance segmenation.
def get_masks(fn, predictor):
    im = cv2.imread(str(fn))
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int) 
    for mask in pred_masks:
        mask = mask * (1-used)
        if mask.sum() >= MIN_PIXELS[pred_class]: # skip predictions with small area
            used += mask
            res.append(rle_encode(mask))
    return res

# Set path.
dataDir=Path('C:/Users/Admin/detectron2/Sartorius-dataset/sartorius-cell-instance-segmentation')

# Get dataset of test images.
ids, masks=[],[]
test_names = (dataDir/'test').ls()

# Get configuration for testing that replicates what was used in the training phase.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.INPUT.MASK_FORMAT='bitmask'
# Set number of clasifications.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# Get trained model for testing.  
cfg.MODEL.WEIGHTS = os.path.join('C:/Users/Admin/detectron2/Sartorius-dataset/output/', "model_final.pth")  
# Set max number of predictions to prevent looping or endless detections.
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
# Instantiate predictor
predictor = DefaultPredictor(cfg)
# Set thresholds per clasification.
THRESHOLDS = [.15, .35, .55]
# Set minimum pixels.
MIN_PIXELS = [75, 150, 75]

# Print all images that the trained model has been tested on. This list can vary depending on images added to the test folder. 
for fn in test_names:
    encoded_masks = get_masks(fn, predictor)
    _, axs = plt.subplots(1,2, figsize=(40,15))
    axs[1].imshow(cv2.imread(str(fn)))
    for enc in encoded_masks:
        dec = rle_decode(enc)
        axs[0].imshow(np.ma.masked_where(dec==0, dec))
        ids.append(fn.stem)
        masks.append(enc)
        
# Save results to submission.csv for kaggle competition submission.
pd.DataFrame({'id':ids, 'predicted':masks}).to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()