import os
import sys
import numpy as np
import cv2
from rgbd import rgbd
import easygui

from matcher import svm_trainer
from matcher import RandomForestTrainer
from matcher import DecisionTreeTrainer
#path=sys.argv[1]

try:
	data=np.load("../database/data.npy")
	labels=np.load("../database/labels.npy")
	print "data and labels are found. Do you want to train model directly"
	ans=raw_input("Y/N ?")
	if ans=="Y":
		print "training svm "
		svm_trainer(data,labels)
		print "svm training completed"
		print "training Decision Tree"
		DecisionTreeTrainer(data,labels)
		print "decision tree training completed"
		print "trainig random forest"
		RandomForestTrainer(data,labels)
		print "Random Forest Training completed"
		exit()
except Exception as E:
	print E
	print "need to read files"
path=easygui.diropenbox("select the root of database")
naming=[]
labels=[]
locations=[]
data=[]
k=0
persons=os.listdir(path)
try:
	persons=map(int,persons)
except Exception as E:
	print E
persons=sorted(persons)
persons=map(str,persons)
for folders in persons:  #loop over all the persons
	folder=path+'/'+folders
	naming.append(folders)
	depth_dir=folder+'/Depth'
	color_dir=folder+'/RGB'
	for images in os.listdir(color_dir):
		color_image=color_dir+'/'+images
		depth_image=depth_dir+'/'+images
		print "checking for ",color_image,
		color=cv2.imread(color_image)
		color=cv2.resize(color,(140,140))
		try:
			depth=cv2.imread(depth_image)
			depth=cv2.resize(depth,(140,140))
		except:
			print "depth image not found for this rgb image ",color_image
			continue
		try:

			feature=rgbd(color,depth).flatten()

			print "feature shape saved by database is ",feature.shape
			data.append(feature)
			#print image
			locations.append(color_image)
			labels.append(k)
		except Exception as e:
			print e
			print "failed for ",color_image
		#data.append(hist(calcgrad(cv2.resize(cv2.imread(image,0),(240,240)))))
	k+=1

np.save("../database/naming",np.array(naming))
np.save("../database/labels",np.array(labels))
np.save("../database/locations",np.array(locations))
np.save("../database/data",np.array(data))
RandomForestTrainer(data,labels)
svm_trainer(data,labels)
DecisionTreeTrainer(data,labels)
