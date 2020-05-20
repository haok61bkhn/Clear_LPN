import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from data.base_dataset import get_transform
from PIL import Image
from models.networks import *
import torch
import cv2
from util.util import *

class Clear_IMG():
    def __init__(self):
        opt = TestOptions().parse() 
        opt.no_flip = True    
        opt.display_id = -1   
        self.model = create_model(opt)     
        self.model.setup(opt)
        self.transform=get_transform(opt)          
        # self.model.eval()
    def clear(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        data=self.transform(img)
        data=data.reshape((1,3,24,24))
        self.model.set_input(data)  # unpack data from data loader
        self.model.test()           # run inference
        visuals = self.model.get_current_visuals()
        image_numpy=[]
        for label, im_data in visuals.items():
            image_numpy=tensor2im(im_data)
            return image_numpy
        return None

if __name__ == '__main__':
    X=Clear_IMG()
    
    img=cv2.imread("test.png")
    cv2.imshow("origin",img)
    img=X.clear(img)
    cv2.imshow("Gan1",img)
    img=X.clear(img)
    cv2.imshow("Gan2",img)
    img=X.clear(img)
    cv2.imshow("Gan3",img)
    img=X.clear(img)
    cv2.imshow("Gan4",img)
    img=X.clear(img)
    cv2.imshow("Gan5",img)
    img=X.clear(img)
    cv2.imshow("Gan6",img)
    img=X.clear(img)
    cv2.imshow("Gan7",img)
    cv2.waitKey(0)
    
    