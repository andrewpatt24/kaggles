from scipy import stats as st
import pandas as pd
import numpy as np
import warnings
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
warnings.simplefilter("ignore")

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


#kobe_id = pd.read_csv('data/kobe_x_id.csv')
kobe_x = pd.read_csv('data/kobe_x_transformed.csv')
kobe_y = pd.read_csv('data/kobe_y.csv')
kobe_shots = pd.DataFrame(pd.read_csv('data/kobe_x.csv')['shot_type_2PT Field Goal'])

print kobe_shots.head()
print kobe_shots[kobe_shots['shot_type_2PT Field Goal']==0].count()
print kobe_shots[kobe_shots['shot_type_2PT Field Goal']==1].count()
score = []
shot_yn_2 = []
shot_yn_3 = []
bnb_2 = BernoulliNB()
bnb_3 = BernoulliNB()

for x, y, s in zip(kobe_x.iterrows(),kobe_y.iterrows(),kobe_shots.iterrows()):
    if s[1]['shot_type_2PT Field Goal'] == 1:
        if x[0]==0:
            print x, y, s
            op = [x[0]+1,0.5]
            score.append(op)
        else:
            if pd.notnull(y[1]['shot_made_flag']):
                bnb_2.partial_fit([x[1].tolist()],[y[1]['shot_made_flag']],classes=[0,1])
                shot_yn_2.append(y[1]['shot_made_flag'])
            else:
                op = [x[0]+1,bnb_2.predict_proba(x[1])[0][1]]
                if x[0]%1000==0:
                    print op
                score.append(op)
    else:
        if x[0]==0:
            print x, y, s
            op = [x[0]+1,0.5]
            score.append(op)
        else:
            if pd.notnull(y[1]['shot_made_flag']):
                bnb_3.partial_fit([x[1].tolist()],[y[1]['shot_made_flag']],classes=[0,1])
                shot_yn_3.append(y[1]['shot_made_flag'])
            else:
                op = [x[0]+1,bnb_3.predict_proba(x[1])[0][1]]
                if x[0]%1000==0:
                    print op
                score.append(op)

#print score[0:5]
with open('data/attempt_3_output.csv','w') as f:
	f.write('shot_id,shot_made_flag'+'\n')
	f.writelines(str(s[0])+','+str(s[1])+'\n' for s in score)

