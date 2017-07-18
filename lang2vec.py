#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals

import argparse, json, sys, itertools
import numpy as np

''' 
A convenience script for accessing the values inside the URIEL typological and geodata knowledge bases 

Author: Patrick Littell
Last modified: July 15, 2016
'''

LETTER_CODES_FILE = "letter_codes.json"
FEATURE_SETS = {
    
    "syntax_wals" : ( "features.npz", "WALS", "S_" ),
    "phonology_wals": ( "features.npz", "WALS", "P_" ),
    "syntax_sswl" : ( "features.npz", "SSWL", "S_" ),
    "syntax_ethnologue": ( "features.npz", "ETHNO", "S_" ),
    "phonology_ethnologue" : ( "features.npz", "ETHNO", "P_" ),
    "inventory_ethnologue" : ( "features.npz", "ETHNO", "INV_" ),
    "inventory_phoible_aa" : ( "features.npz", "PHOIBLE_AA", "INV_" ),
    "inventory_phoible_gm" : ( "features.npz", "PHOIBLE_GM", "INV_" ),
    "inventory_phoible_saphon" : ( "features.npz", "PHOIBLE_SAPHON", "INV_"),
    "inventory_phoible_spa" : ( "features.npz", "PHOIBLE_SPA", "INV_" ),
    "inventory_phoible_ph" : ( "features.npz", "PHOIBLE_PH", "INV_" ),
    "inventory_phoible_ra" : ( "features.npz", "PHOIBLE_RA", "INV_" ),
    "inventory_phoible_upsid" : ( "features.npz", "PHOIBLE_UPSID", "INV_" ),
    "syntax_knn" : ( "feature_predictions.npz", "predicted", "S_" ),
    "phonology_knn" : ( "feature_predictions.npz", "predicted", "P_" ),
    "inventory_knn" : ( "feature_predictions.npz", "predicted", "INV_" ),
    "syntax_average" : ( "feature_averages.npz", "avg", "S_" ),
    "phonology_average" : ( "feature_averages.npz", "avg", "P_" ),
    "inventory_average" : ( "feature_averages.npz", "avg", "INV_" ),
    "fam" : ( "family_features.npz", "FAM", ""),
    "geo" : ( "geocoord_features.npz", "GEOCOORDS", ""),
    
}




LETTER_CODES = {}
with open(LETTER_CODES_FILE, 'r') as file:
    LETTER_CODES = json.load(file)

def get_language_code(lang_code, feature_database):
    # first, normalize to an ISO 639-3 code
    if lang_code in LETTER_CODES:
        lang_code = LETTER_CODES[lang_code]
    if lang_code not in feature_database["langs"]:
        print("ERROR: Language " + lang_code + " not found.", file=sys.stderr)
        sys.exit(2)
    return lang_code

def get_language_index(lang_code, feature_database):
    return np.where(feature_database["langs"] == lang_code)[0][0]

def get_source_index(source_name, feature_database):
    return np.where(feature_database["sources"] == source_name)[0]

def get_feature_names(feature_name_prefix, feature_database):
    return [ f for f in feature_database["feats"] if f.startswith(feature_name_prefix) ]

def get_feature_index(feature_name, feature_database):
    return np.where(feature_database["feats"] == feature_name)[0][0]
    
def get_id_set(lang_codes):
    feature_database = np.load("family_features.npz")
    lang_codes = [ get_language_code(l, feature_database) for l in lang_codes ]
    all_languages = list(feature_database["langs"])
    feature_names = [ "ID_" + l.upper() for l in all_languages ]
    values = np.zeros((len(lang_codes), len(feature_names)))
    for i, lang_code in enumerate(lang_codes):
        feature_index = get_language_index(lang_code, feature_database)
        values[i, feature_index] = 1.0
    return feature_names, values
    
    
def get_named_set(lang_codes, feature_set):
    if feature_set == 'id':
        return get_id_set(lang_codes)
    
    if feature_set not in FEATURE_SETS:
        print("ERROR: Invalid feature set " + feature_set, file=sys.stderr)
        sys.exit()
        
    filename, source, prefix = FEATURE_SETS[feature_set]
    feature_database = np.load(filename)
    lang_codes = [ get_language_code(l, feature_database) for l in lang_codes ]
    lang_indices = [ get_language_index(l, feature_database) for l in lang_codes ]
    feature_names = get_feature_names(prefix, feature_database)
    feature_indices = [ get_feature_index(f, feature_database) for f in feature_names ]
    source_index = get_source_index(source, feature_database)
    feature_values = feature_database["data"][lang_indices,:,:][:,feature_indices,:][:,:,source_index]
    feature_values = feature_values.squeeze(axis=2)
    return feature_names, feature_values

def get_union_sets(lang_codes, feature_set_str):
    feature_set_parts = feature_set_str.split("|")
    feature_names, feature_values = get_named_set(lang_codes, feature_set_parts[0])
    for feature_set_part in feature_set_parts[1:]:
        more_feature_names, more_feature_values = get_named_set(lang_codes, feature_set_part)
        if len(feature_names) != len(more_feature_names):
            print("ERROR: Cannot perform elementwise union on feature sets of different size")
            sys.exit(0)
        feature_values = np.maximum(feature_values, more_feature_values)
    return feature_names, feature_values
    
def get_concatenated_sets(lang_codes, feature_set_str):
    feature_set_parts = feature_set_str.split("+")
    feature_names = []
    feature_values = np.ndarray((len(lang_codes),0))
    for feature_set_part in feature_set_parts:
        more_feature_names, more_feature_values = get_union_sets(lang_codes, feature_set_part)
        feature_names += more_feature_names
        feature_values = np.concatenate([feature_values, more_feature_values], axis=1)
    return feature_names, feature_values
    
def get(languages, feature_set_str, header=False, random=False, minimal=False):
    
    lang_codes = languages.split()
    feature_names, feature_values = get_concatenated_sets(lang_codes, feature_set_str)
    feature_names = np.array([ f.replace(" ","_") for f in feature_names ])
    feats = {}

    if minimal:
        mask = np.all(feature_values == 0.0, axis=0)
        mask |= np.all(feature_values == 1.0, axis=0)
        mask |= np.all(feature_values == -1.0, axis=0)
        unmasked_indices = np.where(np.logical_not(mask))
    else:
        unmasked_indices = np.where(np.ones(feature_values.shape[1]))
        
    if random:
        feature_values = np.random.random(feature_values.shape) >= 0.5
        
    if header:
        print("\t".join(['CODE']+list(feature_names[unmasked_indices])))
    feat_names = feature_names[unmasked_indices]

    for i, lang_code in enumerate(lang_codes):
        values = feature_values[i,unmasked_indices].ravel()
        #values = [ '--' if f == -1 else ("%0.4f"%f).rstrip("0").rstrip(".") for f in values ]
        feats[lang_code] = values
        #print("\t".join([lang_code]+values))
    return feats, feat_names

#if __name__ == '__main__':
#    argparser = argparse.ArgumentParser()
#    argparser.add_argument("languages", default='', help="The languages of interest, in ISO 639-3 codes, separated by spaces (e.g., \"deu eng fra swe\")")
#    argparser.add_argument("feature_set", default='', help="The feature set or sets of interest (e.g., \"syntax_knn\" or \"fam\"), joined by concatenation (+) or element-wise union (|).")
#    argparser.add_argument("-f", "--fields", default=False, action="store_true", help="Print feature names as the first row of data.")
#    argparser.add_argument("-r", "--random", default=False, action="store_true", help="Randomize all feature values (e.g., to make a control group).")
#    argparser.add_argument("-m", "--minimal", default=False, action="store_true", help="Suppress columns that are all 0, all 1, or all nulls.")
#    args = argparser.parse_args()
#    get(args.languages, args.feature_set, args.fields, args.random, args.minimal)
