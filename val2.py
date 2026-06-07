
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR("best.pt")

    model.val(data="data.yaml",
              split='val',
              imgsz=640,
              batch=4,
              save_json=False, 
              # if you need to cal coco metrice
              workers=4,
              device='3',
              project='runs/test',
              name='20251013',     
              )