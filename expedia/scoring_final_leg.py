import pandas as pd 

from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')


chunksize = 1000

mclf = joblib.load('models/model.pkl')
print 'model loaded'

test_x = pd.read_csv('data/final_leg.csv')
print test_x.head()
test_id = test_x['id']
print test_id.head()
test_x.drop('id',1).to_csv('data/data/complete_test_x_vars_final_leg.csv',index=False)
print 'got test ids'
test_x = None

def get_top_x(probs,n=5):
    X_te_op = []
    for p in xrange(len(probs[1])):
        tt = [(x[p].tolist()[1],i) for i, x in enumerate(probs)]
        X_te_op.append([x[1] for x in sorted(tt,reverse=True)][:n])
    return X_te_op

data = pd.read_csv('data/complete_test_x_vars_final_leg.csv',iterator=True,chunksize=chunksize)
print "data read"

def create_kaggle_output(test_id,preds):
    return str(test_id)+','+' '.join([str(x) for x in preds])+'\n'

def create_kaggle_output_lines(test_id,preds):
    str_output = ''
    for i, tid in enumerate(test_id.tolist()):
        str_output += create_kaggle_output(tid,preds[i])
    return str_output

with open('data/kaggle_expedia_submission_test_final_leg.csv','w') as f:
    #f.write('id,hotel_cluster\n')
    for i, chunk in enumerate(data):	
        tpred = mclf.predict_proba(chunk.fillna(-1000))
        #print tpred
        top_x = get_top_x(tpred,5)
        #print top_x
        str_output = create_kaggle_output_lines(test_id[(i)*chunksize:((i)*chunksize+len(top_x))],top_x)
        if i % 100==0:
            print i, 'hthousand'
        f.write(str_output)
