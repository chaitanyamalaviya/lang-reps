import os
import random
random.seed(323323)
source_dir = "/projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok_bpe/"

for lang in os.listdir(source_dir):
  src_combined_filename = "train" + "_" + lang + "_en."+ lang + ".tok.bpe.txt"
  tgt_combined_filename = "train" + "_" + lang + "_en.en" + ".tok.bpe.txt"

  if not os.listdir(source_dir+lang): continue

  ### SOURCE

  lang_tok = "_opt_src_"+lang+" "
  with open(source_dir+lang+"/"+src_combined_filename) as f:
    doc = f.read()
    doc_tok = "\n".join([lang_tok+line for line in doc.split("\n")[:min(len(doc.split("\n")),100)]])

  with open('/projects/tir1/users/cmalaviy/multi-nmt/test-lhs-100/test_'+lang+'_en.'+lang+'.txt','w') as f:
    f.write(doc_tok)

  ### TARGET

  with open(source_dir+lang+"/"+tgt_combined_filename) as f:
    doc = f.read()
    doc_tok = "\n".join([line for line in doc.split("\n")[:min(len(doc.split("\n")),100)]])
  with open('/projects/tir1/users/cmalaviy/multi-nmt/test-lhs-100/test_'+lang+'_en.en.txt','w') as f:
    f.write(doc_tok)
