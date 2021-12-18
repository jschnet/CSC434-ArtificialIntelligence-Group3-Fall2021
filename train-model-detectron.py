#########################################################################################################
#
#   This file's purpose is to train a model against the Sartorius Cell dataset.
#
#########################################################################################################
#
#   The code in this file is based on :-
#   https://www.kaggle.com/slawekbiel/positive-score-with-detectron-2-3-training
#
#########################################################################################################

# Import tools.
# Ignore unused warning, we are using function calls beneath imported directories.import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
# Get scores for accuracy of True positives vs False negatives in order to check loss.
def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
        return np.mean(prec)
    
# Evaluate mapping of segmentation to get True prositive, false positive, and false negative.
class MAPIOUEvaluator(DatasetEvaluator):
    # Initialize data from dataset to be evaluated.
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
         
    # Reset scores.
    def reset(self):
        self.scores = []

    # Process scores
    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    # Evaluate scores.
    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}

# Builder class for trainer to utilize custom evalutor.
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

################################
#   Beginning of program.
################################

# Silly issue that ends child processes requires main function to check if name is euqivalent to main in order to run.
if __name__ == '__main__':
    # Set up logger, path, and retrieve train and validation data that is in json format for coco tools.
    setup_logger()
    dataDir=Path('C:/Users/Admin/detectron2/Sartorius-dataset/sartorius-cell-instance-segmentation')
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train',{}, 'C:/Users/Admin/detectron2/Sartorius-dataset/annotations_train.json', dataDir)
    register_coco_instances('sartorius_val',{},'C:/Users/Admin/detectron2/Sartorius-dataset/annotations_val.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    # Set data for training and build visualizer for computer vision processing.
    d = train_ds[42]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(figsize = (20,15))
    plt.imshow(out.get_image()[:, :, ::-1])
    
    # Get configuration from pre-trained model for segmentation and use configuration for training against validation data set.
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)
    # Create child processes for comparing data during training.
    cfg.DATALOADER.NUM_WORKERS = 2
    # Set weights from best checkpoint model.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # Set images per batch for training against and other parameters for training.
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005 
    cfg.SOLVER.MAX_ITER = 1000    
    cfg.SOLVER.STEPS = []        
    # Header for anchoring.. currently being overwritten by checkpoint.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    # Set total number of clasifications. 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # Set threshold for pixel segmentation.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    # Save data to memory for testing trained data.
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    # Begin training process
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Create saved data from trained model to perform testing in second phase of this project.
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    
    # Pull data from trained samples and prediction segmentations.
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get('sartorius_val')
    # Add data to list.
    outs = []
    # From list pull three images at random to be displayed for visual reference on human end. 
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                   metadata = MetadataCatalog.get('sartorius_train'), 
                    
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
        out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('sartorius_train'))
        out_target = visualizer.draw_dataset_dict(d)
        outs.append(out_pred)
        outs.append(out_target)
        _,axs = plt.subplots(len(outs)//2,2,figsize=(40,45))
    for ax, out in zip(axs.reshape(-1), outs):
        ax.imshow(out.get_image()[:, :, ::-1])
        

        
        