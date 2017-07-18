from __future__ import division
import numpy as np
import pickle

file = open("syntax_avg_lang_knn-2",'r') 
sys_1 = pickle.load(file)
preds_1 = [item for sublist in sys_1 for item in sublist]

file = open("syntax_avg_lang_preds",'r')
sys_2 = pickle.load(file)
preds_2 = [item for sublist in sys_2 for item in sublist]

file = open("syntax_avg_lang_refs",'r')
references = pickle.load(file)
refs = [item for sublist in references for item in sublist]


assert len(preds_1)==len(refs)
assert len(preds_2)==len(refs)

bootstrap_number=10000
count_win_1 = 0
count_win_2 = 0
count_ties = 0


for k in range(bootstrap_number):
    # Make random subset of half sentences in test data
    subset = np.random.choice(len(refs),size=int(0.5*len(refs)))
    #print([preds_1[idx] for idx in subset])
    #print([preds_2[idx] for idx in subset])
    #print([refs[idx] for idx in subset])
    b_1 = sum([1 for idx in subset if preds_1[idx]==refs[idx]])
    b_2 = sum([1 for idx in subset if preds_2[idx]==refs[idx]])
    #print(b_1,b_2) 
    if b_1 > b_2:
	count_win_1 += 1
    elif b_1 < b_2:
	count_win_2 += 1
    else:
	count_ties += 1

print('Win probabilities: %.3f , %.3f , Tie Probability: %.3f' % ((count_win_1/bootstrap_number)*100.0,(count_win_2/bootstrap_number)*100.0, (count_ties/bootstrap_number)*100.0))
