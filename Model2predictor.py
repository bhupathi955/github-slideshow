#!C:/Users/bhupa/Anaconda3/envs/green/python.exe


from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.applications.vgg16 import preprocess_input
from keras.models import model_from_json
import h5py
from PIL import Image
import os, sys
import numpy as np





print("")

dest1 ='C:/xampp/htdocs/rup/proj_imgs/'


json_file = open('green2.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("green2.h5")


loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

imgs = ImageDataGenerator()

batches = imgs.flow_from_directory(
        dest1,
        target_size=(150, 150),
        shuffle=False,
        batch_size=1,
        class_mode=None)


files = os.listdir(dest1+'/0')

count = 0

for f in files:
	if(f != '.' and f != '..'):
		count = count+1


probs = loaded_model.predict_generator(batches,count)


x= probs.tolist()[0]
c1=0



probs.tolist()
Quality = ['Cercospora', 'Common Rust','Good']

for i in range(0,len(x)):
	if(x[i] == 1):
		c1 = i



print(Quality[c1])




