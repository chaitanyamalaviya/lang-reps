from __future__ import print_function, division
from sklearn import preprocessing, neighbors, linear_model, multioutput
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import argparse
import numpy as np
import lang2vec
import os, sys, csv
import pickle

# Load features from URIEL
#all_feat_avgs = np.load("feature_averages.npz")
#lang_codes = list(all_feat_avgs["langs"])

parser = argparse.ArgumentParser()
parser.add_argument("--random", action='store_true')
parser.add_argument("--majority", action='store_true')
parser.add_argument("--family", action='store_true')
parser.add_argument("--results_path", default="syntax_avg_new.csv", type=str)
parser.add_argument("--bible_langs_path", default="bible_langs.txt", type=str)
args = parser.parse_args()

with open(args.bible_langs_path) as f:
  lines = f.readlines()

lang_codes = sorted([line.strip()[-3:] for line in lines])

#Pick feature class
feat_name = "phonology_average"
typ_feats, typ_feat_names = lang2vec.get(" ".join(lang_codes), feat_name)

syntax_avg, syntax_avg_feat_names = lang2vec.get(" ".join(lang_codes), "syntax_average")
phonology_avg, phonology_avg_feat_names = lang2vec.get(" ".join(lang_codes), "phonology_average")
inventory_avg, inventory_avg_feat_names = lang2vec.get(" ".join(lang_codes), "inventory_average")

syntax_knn, syntax_knn_feat_names = lang2vec.get(" ".join(lang_codes), "syntax_knn")
phonology_knn, phonology_knn_feat_names = lang2vec.get(" ".join(lang_codes), "phonology_knn")
inventory_knn, inventory_knn_feat_names = lang2vec.get(" ".join(lang_codes), "inventory_knn")

geog, geog_feat_names = lang2vec.get(" ".join(lang_codes), "geo")
fam, fam_feat_names = lang2vec.get(" ".join(lang_codes), "fam")

# Load language vectors
lang_vecs = np.load("lang_vecs_bible.npy").item()
cell_state_values = np.load("langs_orig_mod_cell.npy").item()


# Get kNN results

geogen = []
geogen_feats = {}
with open("chaitanya-extract/result/geogen-3.csv") as f:
        reader = csv.reader(f)
        for row in reader:
                geogen.append(row)

for g in geogen[1:]:
        geogen_feats[g[0]] = [float(elem) for elem in g[1:]]

# Find responsible hidden node
#print(syntax_avg_feat_names)
#idx = np.where(syntax_avg_feat_names=="S_OBJECT_AFTER_VERB")[0][0]
#print(idx)



logreg_scores = []
nn_scores = []
random_scores = []
scores = []
scores_ref = []
iterate = typ_feats['fra'].shape[0]
all_preds = []
all_refs = []
all_randoms = []

for feat in range(iterate):
  scores_dict = {}
  scores_dict_ref = {}
  preds = []
  refs = []
  # Find langs for which feat exists
  f =  np.asarray([l[1][feat] for l in sorted(typ_feats.items())])
  lang_indices = np.where(f!=-1)   
  feat_langs = [lang_codes[l] for l in list(lang_indices[0])]
  if len(feat_langs)==0:
    print("No languages contain a value for feature %s. Skipping..." % typ_feat_names[feat])
    continue

  # Create feature combinations

  #X_train = np.zeros((len(feat_langs), 512+len(geogen_feats['fra'])))
  #X_train = np.zeros((len(feat_langs), 1024+len(geo['fra'])))
  #X_train = np.zeros((len(feat_langs), 1024))
  #X_train = np.zeros((len(feat_langs), len(geo_feats['fra'])))
  X_train = np.zeros((len(feat_langs), 1024 + 512 + len(geogen_feats['fra'])))
  #X_train = np.zeros((len(feat_langs), len(geogen_feats['fra'])))
  #X_train = np.zeros((len(feat_langs), 512))
  #X_train = np.zeros((len(feat_langs), 512+1024))
  #X_train = np.zeros((len(feat_langs), 1024+len(geogen_feats['fra'])))
  y_train = np.zeros((len(feat_langs)))
  
  for i,l in enumerate(list(lang_indices[0]), start=0):
    y_train[i] = f[l]
    #X_train[i] = lang_vecs['optsrc'+lang_codes[l]]
    #X_train[i] = np.concatenate((lang_vecs['optsrc'+lang_codes[l]],geogen_feats[lang_codes[l]]), axis=0)
    #X_train[i] = cell_state_values[lang_codes[l]][1]
    #X_train[i] = np.concatenate((cell_state_values[lang_codes[l]][0], geog[lang_codes[l]]), axis=0)
    #X_train[i] = geo_feats[lang_codes[l]]
    #X_train[i] = np.concatenate((cell_state_values[lang_codes[l]][0], geogen_feats[lang_codes[l]]))
    #X_train[i] = np.concatenate((cell_state_values[lang_codes[l]][0], lang_vecs['optsrc'+lang_codes[l]]), axis=0)
    X_train[i] = np.concatenate((cell_state_values[lang_codes[l]][0], lang_vecs['optsrc'+lang_codes[l]], geogen_feats[lang_codes[l]]), axis=0)

  #X_train, y_train = shuffle(X_train, y_train)
  lab_enc = preprocessing.LabelEncoder()
  y_train = lab_enc.fit_transform(y_train)

  X_test = X_train[int(len(feat_langs)*.8):]
  y_test = y_train[int(len(feat_langs)*.8):]
  X_train_f = X_train[:int(len(feat_langs)*.8)]
  y_train_f = y_train[:int(len(feat_langs)*.8)]


  # Get random classifier results
  if args.random:
    cv_scores = []
    for i in range(10):
      #X_train, y_train = shuffle(X_train, y_train)
      y_test = y_train[int(len(feat_langs)*.8):]
      y_train_f = y_train[:int(len(feat_langs)*.8)]
      counts = np.bincount(y_train_f)
      y_major = np.argmax(counts)
    
      if np.bincount(y_test).shape[0] > y_major:
        cv_scores.append(np.bincount(y_test)[y_major]/y_test.shape[0])
    
    #print("Random score for feature:", sum(cv_scores[:-1])/len(cv_scores))
    scores_dict['feature'] = typ_feat_names[feat]
    scores_dict['random_score'] = sum(cv_scores[:-1])/len(cv_scores)
    scores.append(scores_dict)
    random_scores.append(scores_dict['random_score'])
    continue

  if np.all(y_train==1.0) or np.all(y_train==0.0):
    print("Feature %s has only one class!" % typ_feat_names[feat])
    logreg_scores.append(1.0)
    nn_scores.append(1.0)
    scores_dict['feature'] = typ_feat_names[feat]
    scores_dict['logreg_score'] = 1.0
    scores_dict['nn_score'] = 1.0
    scores.append(scores_dict)
    continue
  
  ## Postprocessing to handle missing features

  logistic = linear_model.LogisticRegression(C=2.0)
  #mul_out = multioutput.MultiOutputClassifier(logistic)
  try: 
    y = logistic.fit(X_train_f, y_train_f).predict(X_test)
    counts = np.bincount(y_train_f)
    y_major = np.argmax(counts)
    all_randoms.append([y_major]*len(y_test))
  except: continue
  #correct_langs = [(feat_langs[i],y[i]) for i in range(len(y)) if y[i]==y_test[i]]
  #print(correct_langs) 
  lr_score = [logistic.fit(X_train_f, y_train_f).score(X_test, y_test)]
  try: lr_score = cross_val_score(logistic, X=X_train, y=y_train, cv=10)  
  except: continue
  print("\nLogReg score for feature %s is: %f (+/- %0.2f)" % (typ_feat_names[feat], lr_score.mean(), lr_score.std()))
  logreg_scores.append(lr_score.mean())
  #logreg_scores.append(lr_score)
  
  #best_nodes = [logistic.coef_[i].argsort()[-1:][::-1] for i in range(logistic.coef_.shape[0])]
  #print(best_nodes)
  
  #NN classifier 

  nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
  nn_score = nn.fit(X_train, y_train).score(X_test, y_test) 
  nn_score = cross_val_score(nn, X=X_train, y=y_train, cv=10)
  #except: continue
  print("NN score for feature %s is: %f" % (typ_feat_names[feat], sum(nn_score)/len(nn_score)))
  nn_scores.append(nn_score.mean()) 
  
  #Append scores to dict for writing to file
  scores_dict['feature'] = typ_feat_names[feat]
 
  test_lang_idx = int(len(feat_langs)*.8)
  for k,lang in enumerate(feat_langs[test_lang_idx:]):
    scores_dict[lang] = y[k]
    scores_dict_ref[lang] = y_test[k]
    preds.append(y[k])
    refs.append(y_test[k])

  all_preds.append(preds)
  all_refs.append(refs)

  #scores_dict['logreg_score'] = lr_score
  scores_dict['logreg_score'] = lr_score.mean()
  scores_dict['nn_score'] = nn_score.mean()
  scores.append(scores_dict)  
  scores_ref.append(scores_dict_ref)

# Print accuracy
if not args.random:
  print("\nAverage accuracy using LogReg:",sum(logreg_scores)/len(logreg_scores))
  print("Average accuracy using NN:",sum(nn_scores)/len(nn_scores)) 
else:
  print("\nAverage accuracy using a random classifier:",sum(random_scores)/len(random_scores))


with open('syntax_avg_lang_knn-3', 'wb') as fp:
    pickle.dump(all_preds, fp)

with open('syntax_avg_lang_refs', 'wb') as fp:
    pickle.dump(all_refs, fp)

with open('syntax_avg_lang_randoms', 'wb') as fp:
    pickle.dump(all_randoms, fp)

keys = scores[0].keys()
with open('syntax_avg_lang_preds.csv', 'w') as output_file: 
  dict_writer = csv.DictWriter(output_file, keys)
  if os.stat('syntax_avg_lang_preds.csv').st_size == 0: dict_writer.writeheader() 
  dict_writer.writerows(scores)
  output_file.write("")

