import cv2
from rgbd import rgbd
import numpy as np
import scipy
from sklearn import svm
from sklearn.externals import joblib
from collections import Counter
from sklearn import tree
import sklearn
#from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier



data=np.load("../database/data.npy")
labels=np.load("../database/labels.npy")
locations=np.load("../database/locations.npy")
try:
	#svm_model=svm.SVC(kernel='linear', C=1, gamma=1)
	svm_model_linear=joblib.load("../database/svm_linear.pkl")
	svm_model_poly=joblib.load("../database/svm_poly.pkl")
	svm_model_rbf=joblib.load("../database/svm_rbf.pkl")
	#clf = tree.DecisionTreeClassifier()
	decisionTree_model=joblib.load("../database/DecisionTree.pkl")
	rdf_model=joblib.load("../database/rdf.pkl")
except Exception as E:
	print E
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
def euclidean_matcher(feature):
	distance=[]
	k=0
	for i in data:
		dist=scipy.spatial.distance.euclidean(i,feature)
		#print locations[k],labels[k],dist
		distance.append(dist)
		k+=1
	temp=labels.copy()
	distance,temp=zip(*sorted(zip(distance,temp)))
	if distance[0]==0:
		temp=temp[1:11]
	else:
		temp=temp[:10]
	label=Most_Common(temp)
	return label
def svm_matcher_linear(feature):

	predicted= svm_model_linear.predict(feature)
	return predicted
def svm_matcher_rbf(feature):

	predicted= svm_model_rbf.predict(feature)
	return predicted
def svm_matcher_poly(feature):
	predicted= svm_model_poly.predict(feature)
	return predicted
def DecisionTreeMatcher(feature):
	predicted=decisionTree_model.predict(feature)
	return predicted
def rdf_matcher(feature):
	predicted=rdf_model.predict(feature)
	return predicted

def RandomForestTrainer(data,labels):
	print "training random forest"
	forest = RandomForestClassifier(n_estimators=100, random_state=1)
	#multi_target_forest = sklearn.multioutput.MultiOutputClassifier(forest, n_jobs=-1)
	forest.fit(data, labels)
	print "random forest classifier trained"
	joblib.dump(forest,"../database/rdf.pkl")
	print "random forest trained"
def svm_trainer(data,labels):
	print "training svm classifiers"
	svm_model_1=svm.SVC(kernel='linear', C=1, gamma=1)
	svm_model_2=svm.SVC(kernel='poly', C=1, gamma=1)
	svm_model_3=svm.SVC(kernel='rbf', C=1, gamma=1)

	svm_model_1.fit(data, labels)
	svm_model_1.score(data, labels)
	joblib.dump(svm_model_1,"../database/svm_linear.pkl")

	svm_model_2.fit(data, labels)
	svm_model_2.score(data, labels)
	joblib.dump(svm_model_2,"../database/svm_poly.pkl")

	svm_model_3.fit(data, labels)
	svm_model_3.score(data, labels)
	joblib.dump(svm_model_3,"../database/svm_rbf.pkl")
	print "svm classifiers trained"
def DecisionTreeTrainer(data,labels):
	print "training decision tree"
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(data, labels)
	print "decision tree trained"
	joblib.dump(clf,"../database/DecisionTree.pkl")
