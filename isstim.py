import cv2 as cv
import numpy as np
class Stimmer():
    def __init__(self, initialize):
        self.initialize = initialize[:,:,0]
        self.bench =np.sum(self.initialize)

    def isStim(self, image:np.ndarray):
        b = image[:,:,0]
        return self.bench < .95*np.sum(b)