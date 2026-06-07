import warnings, os

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    
# 多卡训练参考<使用教程.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

#"/home/lihejiang/RTDETR-main/dataset/data.yaml"
#"/home/lihejiang/RTDETR-main/hituav/data.yaml"
#"/home/lihejiang/RTDETR-main/ultralytics/cfg/models/rt-detr/rtdetr-r50-bifpn.yaml"
if __name__ == '__main__':
    model = RTDETR("/home/lihejiang/rtdetrce/zidingyi.yaml")
    # model.load('') # loading pretrain weights
    model.train(data="/home/lihejiang/RTDETR-main/dataset/data.yaml",
                cache=False,
                imgsz=640,
                epochs=250,
                batch=4, 
                workers=4, 
                device='5', 
                # resume='', 
                project='runs/train',
                name='202657visdroneembsafpnwan',     
                )