from nltk.tokenize import word_tokenize
import os
import subprocess

dir_path = "/projects/tir2/users/cmalaviy/bible-corpus/combined_test/"
dir_tok_path = "/projects/tir2/users/cmalaviy/bible-corpus/combined_tok/"

for lang in os.listdir(dir_path):
  lang_tok = "_opt_tgt_" + lang

  if not os.path.exists(dir_tok_path+lang):
    os.makedirs(dir_tok_path + lang)
  
  filename_lang = "train_" + lang + "_en." + lang + ".txt"
  filename_en = "train_" + lang + "_en.en.txt"

  with open(dir_path + lang + "/" + filename_lang) as f:
    doc = f.read()
    all_tok_lines = []
    for line in doc.split("\n"):
      tok_line = (lang_tok + " ".join(word_tokenize(line.decode('utf-8')))).lower()
    all_tok_lines.append(tok_line)

  with open(dir_tok_path + lang + "/" + filename_lang[:-4] + ".tok.txt",'w') as f:
    f.write("\n".join(all_tok_lines))

  with open(dir_path + lang + "/" + filename_en) as f:
    doc = f.read()
    all_tok_lines = []
    for line in doc.split("\n"):
      tok_line = " ".join(word_tokenize(line.decode('utf-8'))).lower()
    all_tok_lines.append(tok_line)

  with open(dir_tok_path + lang + "/" + filename_en[:-4] + ".tok.txt",'w') as f:
    f.write("\n".join(all_tok_lines))
