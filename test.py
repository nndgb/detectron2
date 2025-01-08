from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "D:\\detection\\YCX_detectron2\\ycx_train\\Ycx-data\\data_numbers\\test\\WIN_20221124_10_46_40_Pro.jpg"
on_Image(image_path, predictor)


#video_path = ""
#on_Video(0, predictor)#0 表示调用电脑的摄像头来实时预测
