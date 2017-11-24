import cv2
import numpy as np
import os
import easygui
from collections import Counter
from rgbd import rgbd
import scipy
import traceback
from matcher import euclidean_matcher
from matcher import svm_matcher_linear
from matcher import svm_matcher_poly
from matcher import svm_matcher_rbf
from matcher import rdf_matcher
from matcher import DecisionTreeMatcher
import matplotlib.pyplot as plt


data=np.load("database/data.npy")
naming=np.load("database/naming.npy")
labels=np.load("database/labels.npy")
locations=np.load("database/locations.npy")
f=plt.figure()
ax=f.add_subplot(111)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(5)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)
path=easygui.diropenbox("select the directory you want to check your database on")
for func in [svm_matcher_linear]:
	percentages=[]
	people=[]
	images=[]
	success=0
	fail=0
	total=0
	paths=os.listdir(path)
	
	try:
		paths=map(int,paths)
	except Exception as E:
		print E
	paths=sorted(paths)
	paths=map(str,paths)
	for persons in paths:
		#print "checking ",persons,str(func)
		
		data=[]
		labels=[]
		k=0
		
		
		success_temp=0
		fail_temp=0
		total_temp=0
		goal=persons
		people.append(int(persons))
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
			data.append(feature)
			labels.append(k)
			feature=feature.reshape((1,-1))
			#print feature
			#label=euclidean_matcher(feature)
			label=func(feature)
			#label=DecisionTreeMatcher(feature)
			person=naming[label]
			if person==persons:
				success_temp+=1
			else:
				fail_temp+=1
			total_temp+=1


			#print "checking ",color_image,success,fail,total

		success=success+success_temp
		fail+=fail_temp
		total+=total_temp
		images.append(total_temp)
		print str(func).split()[1],persons,float(success_temp)*100/total_temp,float(success)*100/total
		percentages.append(float(success)*100/total)
		k+=1
	#print func.score(data,labels)
	#images,p=np.histogram(labels,range(int(min(labels)),int(max(labels))+2))
	#p,images=zip(*sorted(zip(p,images)))
	people,percentages,images=zip(*sorted(zip(people,percentages,images)))
	#print people
	#print images
	#print percentages

	ax.plot(people,percentages,label=str(func).split()[1])
	
ax.plot(people,images,label='images')
ax.legend()
ax.set_xlabel('peoples')
ax.set_ylabel('matches with number of images')
ax.set_title('change in % matches with variation in number of images')
f.show()
f.savefig("folder1.jpg")
