import scipy as sp
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.decomposition import PCA

models = {
	'GaussianNB':(GaussianNB(),'predict_proba'),
	'Berno-NB':(BernoulliNB(),'predict_proba'),
	'Perceptron':(Perceptron(),'predict'),
	'SGD hinge loss':(SGDClassifier(loss='hinge'),'predict'),
	'SGD mod huber loss':(SGDClassifier(loss='modified_huber'),'predict_proba'),
	'SGD sq hinge loss':(SGDClassifier(loss='squared_hinge'),'predict'),
	'SGD perceptron loss':(SGDClassifier(loss='perceptron'),'predict'),
	'SGD sq loss':(SGDClassifier(loss='squared_loss'),'predict'),
	'SGD huber loss':(SGDClassifier(loss='huber'),'predict'),
	'SGD eps ins loss':(SGDClassifier(loss='epsilon_insensitive'),'predict'),
	'SGD sq eps loss':(SGDClassifier(loss='squared_epsilon_insensitive'),'predict'),
	'SGD hinge loss':(PassiveAggressiveClassifier(),'predict')
}

def train_and_score_model(model_string, models,x_data,y_data):
	curr_model = models[model_string][0]
	predict_type = models[model_string][1]

	with open('data/'+model_string+'.csv','w') as f:
		## write header
		f.write(','.join(['shot_id','shot_made_flag'])+'\n')

		## cycle through data and either fit or score
		for x, y in zip(x_data.iterrows(),y_data['shot_made_flag']):
			if pd.notnull(y):
				curr_model.partial_fit([x[1]],[y],classes=[0,1])
			else:
				if x[0]==0:
					print [str(x[0]+1),'0.5']
					f.write(','.join([str(x[0]+1),'0.5'])+'\n')
				else:
					if predict_type=='predict_proba':
						prediction = curr_model.predict_proba([x[1]])[0]
						#print type(prediction)
						if isinstance(prediction,np.ndarray):
							op = [str(x[0]+1),str(prediction[1])]
						else:
							op = [str(x[0]+1),str(prediction)]
						print op
						f.write(','.join(op)+'\n')
					else:
						prediction = curr_model.predict([x[1]])[0]
						#print type(prediction)
						if isinstance(prediction,np.ndarray):
							op = [str(x[0]+1),str(prediction[1])]
						else:
							op = [str(x[0]+1),str(prediction)]
						print op
						f.write(','.join(op)+'\n')
	print "model_string scored in file "+'data/'+model_string+'.csv'



def main():
	kobe_x = pd.read_csv('data/kobe_x.csv')
	kobe_y = pd.read_csv('data/kobe_y.csv')

	##Lets cheat a little bit and create a PCA of the x data
	pca_x = pd.DataFrame(PCA(5).fit_transform(kobe_x))

	for model in models:
		train_and_score_model(model,models,pca_x,kobe_y)


if __name__ == '__main__':
	main()
	#for model in models:
	#	print models[model][0]

