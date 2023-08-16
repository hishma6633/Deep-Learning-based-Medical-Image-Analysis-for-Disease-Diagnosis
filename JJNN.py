#!/usr/bin/env python
# coding: utf-8

# In[52]:


import sys
import subprocess

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        __import__(package_name)
    except ImportError:
        return False
    return True

# Check if tensorflow is already installed
if is_package_installed('tensorflow'):
    print("TensorFlow is already installed.")
else:
    # Install TensorFlow
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

# Print TensorFlow version
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Continue with the rest of your code...


# In[53]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import random

from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.__version__)


# In[54]:


import numpy as np
import pandas as pd
import os
from glob import glob

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

BASE_DIR = '../input/chest-xray-pneumonia/chest_xray/'
train_dir = os.path.join(BASE_DIR, 'train/')
val_dir = os.path.join(BASE_DIR, 'val/')
test_dir = os.path.join(BASE_DIR, 'test/')

print('Number of images in training set = ', str(len(glob(train_dir + '*/*'))))
print('Number of images in validation set = ', str(len(glob(val_dir + '*/*'))))
print('Number of images in testing set = ', str(len(glob(test_dir + '*/*'))))


# In[55]:


training_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/train/*/*')
validation_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/val/*/*')


total_files = training_images
total_files.extend(validation_images)
print(f'Total number of images : training_images + validation_images = {len(total_files)}\n')

#spliting 80:20
train_images, val_images = train_test_split(total_files, test_size = 0.2)
print(f'After division of 80:20')
print(f'Total number of training images = {len(train_images)}')
print(f'Total number of validation images = {len(val_images)}')


# In[58]:


tf.io.gfile.makedirs('/kaggle/working/val_dataset/NORMAL/')
tf.io.gfile.makedirs('/kaggle/working/val_dataset/PNEUMONIA/')
tf.io.gfile.makedirs('/kaggle/working/train_dataset/NORMAL/')
tf.io.gfile.makedirs('/kaggle/working/train_dataset/PNEUMONIA/')


# In[60]:


import os
import random
import string
import tensorflow as tf

def generate_random_string(length):
    """Generate a random string of specified length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

train_dst_pneumonia_dir = '/kaggle/working/train_dataset/PNEUMONIA/'
train_dst_normal_dir = '/kaggle/working/train_dataset/NORMAL/'
val_dst_pneumonia_dir = '/kaggle/working/val_dataset/PNEUMONIA/'
val_dst_normal_dir = '/kaggle/working/val_dataset/NORMAL/'

for ele in train_images:
    parts_of_path = ele.split('/')
    file_name = parts_of_path[-1]
    
    if 'PNEUMONIA' == parts_of_path[-2]:
        dst_dir = train_dst_pneumonia_dir
    else:
        dst_dir = train_dst_normal_dir
    
    dst_file_path = os.path.join(dst_dir, file_name)
    
    if not tf.io.gfile.exists(dst_file_path):
        tf.io.gfile.copy(src=ele, dst=dst_file_path)

for ele in val_images:
    parts_of_path = ele.split('/')
    file_name = parts_of_path[-1]
    
    if 'PNEUMONIA' == parts_of_path[-2]:
        dst_dir = val_dst_pneumonia_dir
    else:
        dst_dir = val_dst_normal_dir
    
    dst_file_path = os.path.join(dst_dir, file_name)
    
    if not tf.io.gfile.exists(dst_file_path):
        tf.io.gfile.copy(src=ele, dst=dst_file_path)


# In[61]:


print('Pneumonia x-ray images in training set after split = ',len(os.listdir('/kaggle/working/train_dataset/PNEUMONIA/')))
print('Normal x-ray images in training set after split = ',len(os.listdir('/kaggle/working/train_dataset/NORMAL/')))
print('Pneumonia x-ray images in validation set after split = ',len(os.listdir('/kaggle/working/val_dataset/PNEUMONIA/')))
print('Normal x-ray images in validation set after split = ',len(os.listdir('/kaggle/working/val_dataset/NORMAL/')))
print('Pneumonia x-ray images in test set = ',len(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')))
print('Normal x-ray images in test set = ',len(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/NORMAL')))


# In[62]:


train_dir='/kaggle/working/train_dataset/'
val_dir='/kaggle/working/val_dataset/'
test_dir='../input/chest-xray-pneumonia/chest_xray/test/'

train_normal_dir='/kaggle/working/train_dataset/NORMAL'
train_pneumonia_dir='/kaggle/working/train_dataset/PNEUMONIA'
val_normal_dir='/kaggle/working/val_dataset/NORMAL'
val_pneumonia_dir='/kaggle/working/val_dataset/PNEUMONIA'
train_normal_fnames=os.listdir(train_normal_dir)
train_pneumonia_fnames=os.listdir(train_pneumonia_dir)

print(train_normal_fnames[:10])
print(train_pneumonia_fnames[:10])


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

no_cols=4
no_rows=4

pic_index=0
fig=plt.gcf()
fig.set_size_inches(no_cols*4,no_rows*4)

pic_index+=8

normal_pix=[os.path.join(train_normal_dir,fname) for fname in train_normal_fnames[pic_index-8:pic_index]]
pneumonia_pix=[os.path.join(train_pneumonia_dir,fname) for fname in train_pneumonia_fnames[pic_index-8:pic_index]]

for i,img_path in enumerate(normal_pix+pneumonia_pix):
    sp=plt.subplot(no_rows,no_cols,i+1)
    sp.axis()
    
    img=mpimg.imread(img_path)
    plt.imshow(img,cmap='gray')
    
plt.show()


# # MODEL 1

# **First model which we are going to train is a simple CNN model.**

# In[64]:


model=tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same', input_shape=(180, 180, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy','Precision','Recall'])


# In[65]:


model.summary()


# In[66]:


train_datagen=ImageDataGenerator(rescale=1.0/255,
                                 rotation_range=5,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 )

val_datagen=ImageDataGenerator(rescale=1.0/255)

test_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(train_dir,color_mode="grayscale",target_size=(180,180),batch_size=128,class_mode='binary')

val_generator=val_datagen.flow_from_directory(val_dir,color_mode="grayscale",target_size=(180,180),batch_size=128,class_mode='binary')

test_generator=test_datagen.flow_from_directory(test_dir,color_mode="grayscale",target_size=(180,180),batch_size=128,class_mode='binary')


# In[67]:


history=model.fit(train_generator,validation_data=val_generator,epochs=5,verbose=2)


# In[68]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_precision=history.history['precision']
val_precision=history.history['val_precision']

train_recall=history.history['recall']
val_recall=history.history['val_recall']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_precision, 'r', label='Training precision')
plt.plot(epochs, val_precision, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.figure()

plt.plot(epochs, train_recall, 'r', label='Training recall')
plt.plot(epochs, val_recall, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[69]:


eval_result1 = model.evaluate_generator(test_generator, 624)
print('loss  :', eval_result1[0])
print('accuracy  :', eval_result1[1])
print('Precision :', eval_result1[2])
print('Recall :', eval_result1[3])


# # Model 2 (VGG-16)

# In[70]:


from tensorflow.keras.applications.vgg16 import VGG16

pretrained_model4 = VGG16(input_shape=(180, 180, 3), include_top=False, weights=None)

for layer in pretrained_model4.layers:
    layer.trainable = False


# In[71]:


last_output = pretrained_model4.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model4 = tf.keras.Model(pretrained_model4.input, x)

model4.compile(optimizer=RMSprop(lr=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy', 'Precision', 'Recall'])


# In[74]:


train_datagen2=ImageDataGenerator(rescale=1.0/255,
                                 rotation_range=5,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 )

val_datagen2=ImageDataGenerator(rescale=1.0/255)

test_datagen2=ImageDataGenerator(rescale=1.0/255)

train_generator2=train_datagen2.flow_from_directory(train_dir,target_size=(180,180),batch_size=128,class_mode='binary')

val_generator2=val_datagen2.flow_from_directory(val_dir,target_size=(180,180),batch_size=128,class_mode='binary')

test_generator2=test_datagen2.flow_from_directory(test_dir,target_size=(180,180),batch_size=128,class_mode='binary')


# In[76]:


history4 = model4.fit(train_generator2, validation_data=val_generator2, epochs=5, verbose=2)


# In[77]:


acc4 = history4.history['accuracy']
val_acc4 = history4.history['val_accuracy']

train_precision4 = history4.history['precision']
val_precision4 = history4.history['val_precision']

train_recall4 = history4.history['recall']
val_recall4 = history4.history['val_recall']

loss4 = history4.history['loss']
val_loss4 = history4.history['val_loss']
epochs = range(len(acc4))

plt.plot(epochs, acc4, 'r', label='Training accuracy')
plt.plot(epochs, val_acc4, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, train_precision4, 'r', label='Training precision')
plt.plot(epochs, val_precision4, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.show()

plt.plot(epochs, train_recall4, 'r', label='Training recall')
plt.plot(epochs, val_recall4, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.show()

plt.plot(epochs, loss4, 'r', label='Training Loss')
plt.plot(epochs, val_loss4, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[78]:


eval_result4 = model4.evaluate_generator(test_generator2, 624)
print('loss  :', eval_result4[0])
print('accuracy  :', eval_result4[1])
print('Precision :', eval_result4[2])
print('Recall :', eval_result4[3])


# # Model 3(ResNet50)

# In[79]:


from tensorflow.keras.applications.resnet50 import ResNet50
pretrained_model2 = ResNet50(weights=None, include_top=False, input_shape=(180, 180, 3))

#freazing the trained layers
for layers in pretrained_model2.layers:
    layers.trainable = False
#pretrained_model3.summary()


# In[81]:


last_layer=pretrained_model2.get_layer('conv5_block3_1_relu')
last_output = last_layer.output

x=tf.keras.layers.Flatten()(last_output)
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(256,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(1,activation='sigmoid')(x)

model2=tf.keras.Model(pretrained_model2.input,x)

model2.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
               metrics=['accuracy','Precision','Recall'])
#model2.summary()


# In[82]:


history2=model2.fit(train_generator2,validation_data=val_generator2,epochs=5,verbose=2)


# In[83]:


acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']

train_precision2=history2.history['precision']
val_precision2=history2.history['val_precision']

train_recall2=history2.history['recall']
val_recall2=history2.history['val_recall']

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
epochs = range(len(acc2))

plt.plot(epochs, acc2, 'r', label='Training accuracy')
plt.plot(epochs, val_acc2, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, train_precision2, 'r', label='Training precision')
plt.plot(epochs, val_precision2, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.show()

plt.plot(epochs, train_recall2, 'r', label='Training recall')
plt.plot(epochs, val_recall2, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.show()


# In[84]:


eval_result2 = model2.evaluate_generator(test_generator2, 624)
print('loss  :', eval_result2[0])
print('accuracy  :', eval_result2[1])
print('Precision :', eval_result2[2])
print('Recall :', eval_result2[3])


# # Model 4((Inception Model)

# In[85]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
pretrained_model3=InceptionV3(input_shape=(180,180,3),
                             include_top=False,
                             weights=None)
#freazing the trained layers
for layers in pretrained_model3.layers:
    layers.trainable=False


# In[86]:



last_layer=pretrained_model3.get_layer('mixed10')
last_output = last_layer.output
x=tf.keras.layers.Flatten()(last_output)
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(256,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(1,activation='sigmoid')(x)

model3=tf.keras.Model(pretrained_model3.input,x)

model3.compile(optimizer=RMSprop(lr=0.001),
          loss='binary_crossentropy',
           metrics=['accuracy','Precision','Recall'])
# model4.summary()


# In[87]:


history3=model3.fit(train_generator2,validation_data=val_generator2,epochs=5,verbose=2)


# In[88]:


acc3 = history3.history['accuracy']
val_acc3 = history3.history['val_accuracy']

train_precision3=history3.history['precision']
val_precision3=history3.history['val_precision']

train_recall3=history3.history['recall']
val_recall3=history3.history['val_recall']

loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']
epochs = range(len(acc3))

plt.plot(epochs, acc3, 'r', label='Training accuracy')
plt.plot(epochs, val_acc3, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, train_precision3, 'r', label='Training precision')
plt.plot(epochs, val_precision3, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.show()

plt.plot(epochs, train_recall3, 'r', label='Training recall')
plt.plot(epochs, val_recall3, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.show()

plt.plot(epochs, loss3, 'r', label='Training Loss')
plt.plot(epochs, val_loss3, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[89]:


eval_result3 = model3.evaluate_generator(test_generator2, 624)
print('loss  :', eval_result3[0])
print('accuracy  :', eval_result3[1])
print('Precision :', eval_result3[2])
print('Recall :', eval_result3[3])


# # Model 5 (DenseNet)

# In[90]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.optimizers import RMSprop

# Create the DenseNet model
input_shape = (180, 180, 3)
base_model = DenseNet121(input_shape=input_shape, include_top=False, weights=None)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# Train the model
history = model.fit(train_generator2, validation_data=val_generator2, epochs=5, verbose=2)


# In[91]:


# Plot the training and validation metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, precision, 'r', label='Training precision')
plt.plot(epochs, val_precision, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.show()

plt.plot(epochs, recall, 'r', label='Training recall')
plt.plot(epochs, val_recall, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[92]:


# Evaluate the model on test data
eval_result = model.evaluate_generator(test_generator2, 624)
print('Loss:', eval_result[0])
print('Accuracy:', eval_result[1])
print('Precision:', eval_result[2])
print('Recall:', eval_result[3])


# # Model 6(Xception)

# In[93]:


from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

pretrained_model4 = Xception(input_shape=(180, 180, 3),
                             include_top=False,
                             weights=None)

# Freezing the trained layers
for layer in pretrained_model4.layers:
    layer.trainable = False

last_layer = pretrained_model4.get_layer('block14_sepconv2_act')
last_output = last_layer.output

x = tf.keras.layers.GlobalAveragePooling2D()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model4 = tf.keras.Model(pretrained_model4.input, x)

model4.compile(optimizer=RMSprop(lr=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy', 'Precision', 'Recall'])

history4 = model4.fit(train_generator2, validation_data=val_generator2, epochs=5, verbose=2)


# In[94]:



acc4 = history4.history['accuracy']
val_acc4 = history4.history['val_accuracy']

train_precision4 = history4.history['precision']
val_precision4 = history4.history['val_precision']

train_recall4 = history4.history['recall']
val_recall4 = history4.history['val_recall']

loss4 = history4.history['loss']
val_loss4 = history4.history['val_loss']
epochs = range(len(acc4))

plt.plot(epochs, acc4, 'r', label='Training accuracy')
plt.plot(epochs, val_acc4, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, train_precision4, 'r', label='Training precision')
plt.plot(epochs, val_precision4, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.show()

plt.plot(epochs, train_recall4, 'r', label='Training recall')
plt.plot(epochs, val_recall4, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.show()

plt.plot(epochs, loss4, 'r', label='Training Loss')
plt.plot(epochs, val_loss4, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(epochs, loss4, 'r', label='Training Loss')
plt.plot(epochs, val_loss4, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[95]:


eval_result4 = model4.evaluate_generator(test_generator2, 624)
print('loss  :', eval_result4[0])
print('accuracy  :', eval_result4[1])
print('Precision :', eval_result4[2])
print('Recall :', eval_result4[3])

