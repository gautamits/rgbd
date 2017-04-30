import cv2
import numpy as np
import os
import easygui
from collections import Counter
from rgbd import rgbd
import scipy
import traceback
from matcher import euclidean_matcher
from matcher import svm
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
success=0
fail=0
total=0
data=np.load("database/data.npy")
naming=np.load("database/naming.npy")
labels=np.load("database/labels.npy")
locations=np.load("database/locations.npy")

path=easygui.diropenbox("select the directory you want to check your database on")
for persons in os.listdir(path):
	goal=persons
	color_path=path+'/'+persons+'/RGB'
	for color_images in os.listdir(color_path):
		color_image=color_path+'/'+color_images
		rgb=cv2.imread(color_image)
		rgb=cv2.resize(rgb,(140,140))
		depth_path=color_image.split("/")[::-1]
		
		depth_path[1]="Depth"
		depth_path=depth_path[::-1]
		depth_path="/".join(depth_path)
		#depth_path=depth_path+'/'+color_images
		#print depth_path
		try:
			depth=cv2.imread(depth_path)
			depth=cv2.resize(depth,(140,140))
		except:
			"depth not read properly for ",color_path
			continue
		feature=rgbd(rgb,depth)
		feature=feature.reshape((1,-1))
		#print feature
		#label=euclidean_matcher(feature)
		label=svm(feature)
		person=naming[label]
		if person==persons:
			success+=1
		else:
			fail+=1
		total+=1


		#print "checking ",color_image,success,fail,total
	print "after ",persons," percentage is ",float(success)*100/total

