import numpy as np

class PersonIDType():
    def __init__(self, currCam=-1, label=-1, feats=np.zeros((1,1280)), bbox=np.zeros((1,4)), lock=0):
        self.currCam = currCam
        self.label = label
        self.feats = feats.reshape((1280,1))
        self.featidx = 0
        self.bbox = bbox
        self.lock = lock