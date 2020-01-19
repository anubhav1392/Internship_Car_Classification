'''
Class 0:Honda city,1:Swift Dezire
'''

import tensorflow as tf
import tensorflow.compat.v1 as tc
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,confusion_matrix
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip
)


#Set Path
main_path=r'C:\Users\Anu\Downloads\intrnship\car_images'
car_models=os.listdir(main_path)

#Hyperparameters
BATCH_SIZE=5
DIMS=(128,128,3)
EPOCHS=25

class_dict={car_models[0]:0,car_models[1]:1}

#Now lets import image indexes
image_indexes=[]
targets=[]
for class_,class_name in enumerate(car_models):
    images_ids=os.listdir(os.path.join(main_path,class_name))
    for ix in images_ids:
        image_indexes.append(os.path.join(main_path,class_name,ix))
        targets.append(class_)

#Delete variables that we dont need
del images_ids,ix,class_,class_name

#Shuffle Dataset
image_indexes,targets=shuffle(image_indexes,targets)
#Convert target to arrays
targets=np.array(targets)
targets=np.reshape(targets,(len(targets),1))

#Split Dataset
train_X,val_X,train_y,val_y=train_test_split(image_indexes,targets,test_size=0.15)

def augment_flips_color(p=.5):
    return Compose([
        RandomRotate90(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        Blur(blur_limit=3),
        VerticalFlip(),
        HorizontalFlip()
        
    ], p=p)

def read_image(indexes,is_train=True):
    tmp=np.zeros((BATCH_SIZE,DIMS[0],DIMS[1],DIMS[2])) 
    aug = augment_flips_color(p=1)
    for ix,img_name in enumerate(indexes):
        img=cv2.imread(img_name)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=img.astype('float')/255.
        img=cv2.resize(img,(DIMS[0],DIMS[1]))
        if is_train:
            img = aug(image=img)['image']
        tmp[ix]=img
    return tmp



#Disable Eager Execution
tc.disable_eager_execution()

#Get number of training batches
train_batches=int(len(train_X)//BATCH_SIZE)
val_batches=int(len(val_X)//BATCH_SIZE)
train_loss=[]
val_loss=[]

#Placeholders
model_input=tc.placeholder(tc.float32,[None,DIMS[0],DIMS[1],DIMS[2]],name='image_input')
model_target=tc.placeholder(tc.float32,[None,1],name='class_targets')

#Model
base_feat=tf.keras.applications.ResNet50(weights=r'C:\Users\Anu\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                         include_top=False)(model_input)
x=tc.keras.layers.GlobalAveragePooling2D()(base_feat)
x=tf.keras.layers.Dropout(0.5)(x)
x=tf.keras.layers.Dense(2048,activation='relu')(x)
logits=tf.keras.layers.Dense(1)(x)
probs=tf.nn.sigmoid(logits)

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=model_target,logits=logits))
opt=tc.train.AdamOptimizer(0.0001).minimize(loss)


#Model saver
saver=tc.train.Saver()
saver_path=r'C:\Users\Anu\Downloads\intrnship\output\classifier.ckpt'

with tc.Session() as sess:
    sess.run(tc.global_variables_initializer())
    print('Training started')
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1,EPOCHS))
        for batch_ix in tqdm(range(train_batches)):
            img_ixs=train_X[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            img_ixs=read_image(img_ixs)
            train_labels=train_y[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            t_loss,_=sess.run([loss,opt],feed_dict={model_input:img_ixs,model_target:train_labels})
        
        for batch_ix in tqdm(range(val_batches)):
            img_ixs=val_X[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            img_ixs=read_image(img_ixs,is_train=False)
            val_labels=val_y[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            v_loss=sess.run(loss,feed_dict={model_input:img_ixs,model_target:val_labels})

        print('Training Loss: {}'.format(t_loss))
        print('Validation Loss: {}'.format(v_loss))
        
        train_loss.append(t_loss)
        val_loss.append(val_loss)
        
         #Save model 
        if epoch==0:
            print('Loss improved from {} to {}'.format(0,v_loss))
            saver.save(sess,saver_path)
            tmp_v_loss=v_loss
            
        elif v_loss<tmp_v_loss:
            print('Loss improved from {} to {}'.format(tmp_v_loss,v_loss))
            saver.save(sess,saver_path)
            tmp_v_loss=v_loss
        print('==============================================')
        

#Plot Loss
epochs=range(1,len(train_loss)+1)
plt.plot(epochs,train_loss,'b',color='red',label='Training Loss')
plt.plot(epochs,val_loss,'b',color='blue',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.figure()
plt.show()
            
#Lets evaluate our model on validation dataset
preds=[]
with tf.Session() as sess:
    saver.restore(sess,saver_path)
    for img in tqdm(val_X):
        img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=img.astype('float')/255.
        img=cv2.resize(img,(DIMS[0],DIMS[1]))
        img=np.expand_dims(img,0)
        preds.extend(sess.run(probs,feed_dict={model_input:img}))
        preds=[0 if i<0.5 else 1 for i in preds]

confusion_matrix(val_y,preds)
        
        
    
        