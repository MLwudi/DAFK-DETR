import warnings, os

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR("zidingyi.yaml")
    model.train(data="data.yaml",
                cache=False,
                imgsz=640,
                epochs=250,
                batch=4, 
                workers=4, 
                device='5', 
                # resume='', 
                project='runs/train',
                name='2026',     
                )