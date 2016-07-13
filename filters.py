"""Filters"""

from skimage.color import rgb2gray
from skimage.transform import resize

import numpy as np
from modular_rl import *

class ObFilterFF(object):
    def __init__(self, new_width, new_height):
        self.w = new_width
        self.h = new_height
        self.f = Flatten()

    def __call__(self, ob):
        out = resize(rgb2gray(ob), (self.h, self.w))
        return self.f(out)

    def output_shape(self, input_shape):
        return (self.h * self.w,)

class ObFilterCNN(object):
    def __init__(self, new_width, new_height):
        self.w = new_width
        self.h = new_height

    def __call__(self, ob):
        out = resize(rgb2gray(ob), (self.h, self.w))
        return out.reshape(out.shape + (1,))

    def output_shape(self, input_shape):
        return (self.h, self.w, 1)

class ActFilter(object):
    def __init__(self, lookup):
        self.lookup = lookup
        self.n = len(self.lookup)

    def __call__(self, act):
        action_list = np.zeros(43) # Doom has 43 actions
        action_list[self.lookup[act]] = 1
        return action_list

    def output_shape(self):
        return self.n
