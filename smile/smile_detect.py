import os
import tensorflow as tf
import cv2
import numpy as np
import glob
import PIL

project_root = os.path.dirname(__file__)

class BaseModel():
    def __init__(self,project_root = project_root):
        self.path = os.path.join(os.path.dirname(__file__), 'models')
        self.model = None

    def weights_load(self,filename):
        if not filename.endswith('.h5'):
            filename += '.h5'
        try:
            self.model.load_weights(filename)
            print('pretrained weights loaded')
        except(FileNotFoundError,OSError):
            print(f'weights loading failed :{filename}')

class MiniVGG(BaseModel):
    def __init__(self, blocks = 5):
        super(MiniVGG,self).__init__()
        self.blocks = blocks
        self.model_init()

    def model_init(self):
        tensor = tf.keras.Input(shape = (100,100,1))
        x = tensor

        channels = 16
        for i in range(self.blocks):
            x = tf.keras.layers.Conv2D(channels,3,padding='same',activation='relu')(x)
            x = tf.keras.layers.Conv2D(channels,3,padding='same',activation='relu')(x)
            x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
            channels *= 2
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1,activation='sigmoid')(x)
        model = tf.keras.Model(inputs = tensor,outputs = out)
        self.model = model

# if __name__ == '__main__':
#     # model = MiniVGG()
#     # model.weights_load('\smile\minivgg_weights.h5')

#     # paths = glob()