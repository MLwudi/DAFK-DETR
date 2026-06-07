
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO
if __name__ == '__main__':

#"/home/lihejiang/RTDETR-main/runs/train/2025714all-repconv/weights/best.pt"-repconv    "/home/lihejiang/RTDETR-main/runs/train/2025714all/weights/best.pt"c-all
#"/home/lihejiang/RTDETR-main/runs/train/2025713R18-conv3xc/weights/best.pt"    "/home/lihejiang/RTDETR-main/runs/train/2025713R183/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251010hituavall2/weights/best.pt"
#/home/lihejiang/RTDETR-main/runs/train/20251010hituavr18/weights/best.pt
#"/home/lihejiang/RTDETR-main/runs/train/20251010hituavR50/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251011hituavMODELALL250/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251012visdronemodelallshapeiouhituav/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251013hituavmodelall/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251016hituavmodelallCIOU/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251016hituavmodelallCIOU2/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251016hituavmodelallEIOU2/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251016hituavmodelallSIOU/weights/best.pt"
#"/home/lihejiang/RTDETR-main/dataset/data.yaml"
#"/home/lihejiang/RTDETR-main/hituav/data.yaml"
#"/home/lihejiang/RTDETR-main/runs/train/20251018hituvDIOU/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251019bifpnvisdrone/weights/best.pt"
#"/home/lihejiang/RTDETR-main/runs/train/20251013hituavmodelall/weights/best.pt"
    model = RTDETR("/home/lihejiang/RTDETR-main/runs/distill/rtdetr-cwd20251164/weights/best.pt")
    #model = YOLO("/home/lihejiang/RTDETR-main/runs/train/2025121visdrone/weights/best.pt")
    model.val(data="/home/lihejiang/RTDETR-main/dataset/data.yaml",
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