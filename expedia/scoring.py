import pandas as pd 

from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

mclf = joblib.load('models/model.pkl')
print 'model loaded'

test_x = pd.read_csv('data/complete_test_x.csv')
test_id = test_x['id']

def get_top_x(probs,n=5):
    X_te_op = []
    for p in xrange(len(probs[1])):
        tt = [(x[p].tolist()[1],i) for i, x in enumerate(probs)]
        X_te_op.append([x[1] for x in sorted(tt,reverse=True)][:n])
    return X_te_op

data = pd.read_csv('data/complete_test_x_vars.csv')
print "data read"

def create_kaggle_output(test_id,preds):
    return str(test_id)+','+' '.join([str(x) for x in preds])+'\n'

with open('data/kaggle_expedia_submission_test.csv','w') as f:
    for i, dat in data.iterrows():
    	if i % 10000:
    		print i
        tpred = mclf.predict_proba(dat.fillna(-1000))
        #print tpred
        top_x = get_top_x(tpred,5)
        #print top_x
        str_output = create_kaggle_output(test_id[i],top_x[0])
        f.write(str_output)
