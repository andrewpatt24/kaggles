import scipy as sp
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model, svm
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import itertools


def logloss(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll


def predict_probabilities(X_train,X_test,y_train,threshold,component,m):
	## Selector phase
	#selector = SelectFromModel(linear_model.LogisticRegression(),threshold=threshold)
	#print X_train, y_train
	#selector.fit(X_train,y_train)
	#new_X_train = selector.transform(X_train)
	
	##PCA phase
	pca = PCA(n_components=component)
	
	pca.fit(X_train)
	pca_variance =  sum(pca.explained_variance_ratio_)
	pca_X_train = pca.transform(X_train)
	
	#convert the X_test
	pca_X_test = pca.transform(X_test)
	
	##Model phase
	model = m[1]
	model.fit(pca_X_train,y_train)
	return model.predict_proba(pca_X_test), pca_variance


def model_tests(X_train,X_test,y_train,y_test,thresholds,components,models):
	output = []
	for threshold, component, m in itertools.product(thresholds,components,models):
		#print threshold, component, m
		y_pred, pca_variance = predict_probabilities(X_train,X_test,y_train,threshold,component,m)
		##test phase
		output.append([m, threshold, component, pca_variance, logloss(y_test,[x[1] for x in y_pred])])

	return pd.DataFrame(output,columns = ['model','selection_threshold','N_pca','pca_var','logloss'])


def predict_best_hyper_params(X_train,X_test,y_train,y_test,thresholds,components,models):
	output = test_model(X_train,X_test,y_train,y_test,thresholds,components,models)
	hyper_params = output.sort('logloss',ascending=True).reset_index().head(1).values.tolist()[0]
	return hyper_params


def global_prediction(kobe_x_train,kobe_x_test,kobe_y_train,thresholds,components,models):
	X_train, X_test, y_train, y_test = train_test_split(kobe_x_train, kobe_y_train, test_size=0.33, random_state=42)
	output = model_tests(X_train,X_test,y_train,y_test,thresholds,components,models)
	#print output[output['logloss']>0.01].sort('logloss',ascending=True)
	hyper_params = output[output['logloss']>0.01].sort('logloss',ascending=True).reset_index().head(1).values.tolist()[0]
	#print hyper_params
	probs = predict_probabilities(kobe_x_train,kobe_x_test,kobe_y_train,hyper_params[2],hyper_params[3],hyper_params[1])[0]
	return [x[1] for x in probs]


def x_range(start,end,by):
	while start < end:
		yield start
		start += by


if __name__ == '__main__':
	k_x = pd.read_csv('data/kobe_x.csv')
	k_y = pd.read_csv('data/kobe_y.csv')

	X_scaled = preprocessing.scale(k_x)

	thresholds = [0.05]
	components = [5]

	models = [
		['logmodel',linear_model.LogisticRegression()]#,
		#['svc',svm.SVC(probability=True)]
		]

	with open('data/a5_scoring.csv','w') as f:
		f.write('shot_id,shot_made_flag'+'\n')
		i=0
		dx = []
		dy = []
		predictions = []
		for x, y in zip(X_scaled,k_y['shot_made_flag'].fillna(-1000)):
			if i==0:
				#print x
				#predictions.append([x[0],0.5])
				f.write(','.join([str(z) for z in [i+1,0.5]])+'\n')
				#print x
				print [(i+1),0.5]
			else:
				if y != -1000:
					dx.append(x)
					dy.append(y)
				else:
					if i<=100:
						#predictions.append([x[0],np.mean(dy)])
						f.write(','.join([str(z) for z in [i+1,np.mean(dy)]])+'\n')
						print [i+1,np.mean(dy)]
					else:
						#predictions.append(
						#	[x[0]+1,global_prediction(dx,kobe_x_test,dy,thresholds,components,models)]
						#)
						f.write(','.join(
							[str(z) for z in [i+1,global_prediction(dx,[x],dy,thresholds,components,models)[0]]]
							)+'\n')
						print [i+1,global_prediction(dx,[x],dy,thresholds,components,models)]
			i+=1
			if i>10000000:
				break