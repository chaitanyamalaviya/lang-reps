#!/usr/bin/env python3 

import argparse, itertools, sys
sys.path.append("../lib")
from uriel_lib_data2 import *
from uriel_lib_distance import *
from uriel_lib_predict import *
from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def run(inputFilename, metricFilename, cv=3, neighbors=10, 
            threshold=1.0, gene_coeff=1, gene_exp=1, geo_coeff=1, geo_exp=1, 
            feat_coeff=1, feat_exp=1, verbose=0, limit=-1, labeled=0, jobs=1, weighted=True, algorithm="knn", prefix=""):
        
    
    if verbose: 
        print("Running " + repr(cv) + "-fold cross-validation:")
        
        if algorithm == "knn":
            print("  algorithm = knn")
            print("  neighbors = " + repr(neighbors))
            print("  distance threshold = " + repr(threshold))
            print("  gene_coeff = " + repr(gene_coeff))
            print("  gene_exp = " + repr(gene_exp))
            print("  geo_ceff = " + repr(geo_coeff))
            print("  geo_exp = " + repr(geo_exp))
            print("  feat_coeff = " + repr(feat_coeff))
            print("  feat_exp = " + repr(feat_exp))
            print("  weighted = " + repr(weighted))
        elif algorithm == "random":
            print("  algorithm = random")
        elif algorithm == "zero":
            print("  algorithm = zero")
        elif algorithm == "majority":
            print("  algorithm = majority")
        print("Loading feature dataset")  
        
    inputDataset = Dataset()
    inputDataset.loadFromNPZ(inputFilename)
    inputFeatureList = inputDataset.getFeatureList()
        
    # pick out the features of interest
    inputFeatureList = [((l,f,s),v) for ((l,f,s),v) in inputFeatureList if f.startswith(prefix)]
    
    
    if limit > 0:
        inputFeatureList = inputFeatureList[:limit]
        
    print("Using " + repr(len(inputFeatureList)) + " features")
    
    np.random.shuffle(inputFeatureList)
    inputData, inputTarget = zip(*inputFeatureList)
    inputData = np.array(inputData)
    inputTarget = np.array(inputTarget)
    
        
    if verbose:
        print("Making classifier")
    
    if algorithm == 'knn':
        distanceDataset = DistanceDataset()
        distanceDataset.loadFromNPZ(metricFilename)
        rc = GeoMetricKNNClassifier(distanceDataset, k=neighbors, threshold=threshold, verbose=verbose, 
                                    gene_coeff=gene_coeff, gene_exp=gene_exp, geo_coeff=geo_coeff, geo_exp=geo_exp, weighted=weighted) 
    elif algorithm == 'baggingKNN':
        distanceDataset = DistanceDataset()
        distanceDataset.loadFromNPZ(metricFilename)
        rc = GeoMetricKNNClassifier(distanceDataset, k=neighbors, threshold=threshold, verbose=verbose, 
                                    gene_coeff=gene_coeff, gene_exp=gene_exp, geo_coeff=geo_coeff, geo_exp=geo_exp, weighted=weighted) 
        rc = BaggingClassifier(rc, n_estimators=3, bootstrap=False)
    elif algorithm == 'random':
        rc = RandomClassifier()
    elif algorithm == 'majority':
        rc = AverageClassifier()
    elif algorithm == 'zero':
        rc = ZeroClassifier()
    elif algorithm == 'als':
        geocoordDataset = Dataset()
        geocoordDataset.loadFromNPZ("../../../data/results/geodata/geocoord_features.npz")
        rc = MatrixFactorClassifier(algorithm='als',additionalDatasets=[])
    elif algorithm == 'svd':
        rc = MatrixFactorClassifier(algorithm='svd')
    elif algorithm == "featureKNNdivided":
        rc = DividedFeatureMetricKNNClassifier(k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted)
    elif algorithm == "featureKNN":
        rc = FeatureMetricKNNClassifier(k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted)
    elif algorithm == "KNNdivided":
        distanceDataset = DistanceDataset()
        distanceDataset.loadFromNPZ(metricFilename)
        rc = DividedKNNClassifier(distanceDataset, k=neighbors, threshold=threshold, verbose=verbose, 
                                  gene_coeff=gene_coeff, gene_exp=gene_exp, 
                                  geo_coeff=geo_coeff, geo_exp=geo_exp, 
                                  feat_coeff=feat_coeff, feat_exp=feat_exp,
                                  weighted=weighted) 
    elif algorithm == "ensemble":
        distanceDataset = DistanceDataset()
        distanceDataset.loadFromNPZ(metricFilename)
        rc = EnsembleClassifier()
        c = SavedMetricKNNClassifier(distanceDataset, "GENETIC", k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted) 
        rc.addClassifier(c)
        c = SavedMetricKNNClassifier(distanceDataset, "GEOGRAPHIC", k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted) 
        rc.addClassifier(c)
        c = DividedFeatureMetricKNNClassifier(k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted)
        #c = DividedFeatureMetricKNNClassifier(k=neighbors, threshold=threshold, verbose=verbose, weighted=weighted)
        rc.addClassifier(c)
        #c = MatrixFactorClassifier(algorithm='als')
        #rc.addClassifier(c)
    else:
        sys.exit("ERROR: No algorithm named " + algorithm)
    
    if verbose:
        print("Cross validating")
        
    if labeled:
        labels = [ inputDataset.languageCodes.index(l) for l,_,_ in inputData ]
        kf = cross_validation.LabelKFold(labels, n_folds=cv)
    else:
        kf = cross_validation.KFold(len(inputData), n_folds=cv)
    
    scores = cross_validation.cross_val_score(rc, inputData, inputTarget, cv=kf, scoring="accuracy", n_jobs=jobs, verbose=verbose)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2 * 100))
    
    
  

defaults = {
    "inputFile" : "../../../data/results/features/features.npz",
    "metricFile" : "../../../data/results/distances/distances.npz",
    "cv" : "10",
    "neighbors" : "10",
    "threshold" : "1.0",
    "gene_coeff" : "1",
    "gene_exp" : "1",
    "geo_coeff" : "1",
    "geo_exp" : "1",
    "feat_coeff" : "1",
    "feat_exp" : "1",
    "verbose" : "2",
    "limit" : "-1",
    "labeled" : "0",
    "jobs" : "1",
    "prefix" : ""
}
  
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inputFile", nargs='?', default=[defaults["inputFile"]], help="The CSV file containing the input data.")
    argparser.add_argument("--metricFile", nargs='?', default=[defaults["metricFile"]], help="The CSV file containing the metric data.")
    argparser.add_argument("--cv", nargs='?', default=defaults["cv"], help="The number of folds for k-fold cross-validation.")
    argparser.add_argument("--neighbors", nargs="?", default=defaults["neighbors"], help="The number of neighbors to consider for kNN classification.")
    argparser.add_argument("--threshold", nargs="?", default=defaults["threshold"], help="A distance threshold beyond which neighbors will not be considered.")
    argparser.add_argument("--gene_coeff", nargs="?", default=defaults["gene_coeff"], help="The coefficient of the genetic distance measure.")
    argparser.add_argument("--gene_exp", nargs="?", default=defaults["gene_exp"], help="The exponent of the genetic distance measure.")
    argparser.add_argument("--geo_coeff", nargs="?", default=defaults["geo_coeff"], help="The coefficient of the geographic distance measure.")
    argparser.add_argument("--geo_exp", nargs="?", default=defaults["geo_exp"], help="The exponent of the geographic distance measure.")
    argparser.add_argument("--feat_coeff", nargs="?", default=defaults["feat_coeff"], help="The coefficient of the featural distance measure.")
    argparser.add_argument("--feat_exp", nargs="?", default=defaults["feat_exp"], help="The exponent of the featural distance measure.")
    argparser.add_argument("--verbose", nargs="?", default=defaults["verbose"], help="How verbose to be; higher numbers are more messages.")
    argparser.add_argument("--limit", nargs="?", default=defaults["limit"], help="For testing, limit dataset to this number of observations; <= 0 gives all values.")
    argparser.add_argument("--labeled", nargs="?", default=defaults["labeled"], help="If > 0, no data from the same language will be in both testing and training set.")
    argparser.add_argument("--jobs", nargs="?", default=defaults["jobs"], help="The number of CPUs to use.")
    argparser.add_argument("--weighted", dest="weighting", action="store_true")
    argparser.add_argument("--unweighted", dest="weighting", action="store_false")
    argparser.add_argument("--algorithm", nargs="?", default="knn", help="knn, random, zero, or majority")
    argparser.add_argument("--prefix", nargs="?", default=defaults["prefix"])
    argparser.set_defaults(weighting=True)
    args = argparser.parse_args()

    run(args.inputFile[0], 
        args.metricFile[0], 
        cv=int(args.cv), 
        neighbors=int(args.neighbors), 
        threshold=float(args.threshold), 
        gene_coeff=float(args.gene_coeff), 
        gene_exp=float(args.gene_exp), 
        geo_coeff=float(args.geo_coeff), 
        geo_exp=float(args.geo_exp), 
        feat_coeff=float(args.feat_coeff), 
        feat_exp=float(args.feat_exp), 
        verbose=int(args.verbose),
        limit=int(args.limit),
        labeled=int(args.labeled),
        jobs=int(args.jobs),
        weighted=args.weighting,
        algorithm=args.algorithm,
        prefix=args.prefix)
