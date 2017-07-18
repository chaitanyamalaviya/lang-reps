import os
from subprocess import call

dir_bpe_path = "/projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok_bpe/"

for lang in os.listdir("/projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok/"):
  
  if not os.path.exists(dir_bpe_path + lang):
    os.makedirs(dir_bpe_path + lang)
  
  filename_lang = "/train_" + lang + "_en." + lang + ".tok.txt"
  filename_en = "/train_" + lang + "_en.en.tok.txt"  
  print("python /home/cmalaviy/multi-nmt/subword-nmt/apply_bpe.py -c /projects/tir2/users/cmalaviy/bible-corpus/combined-bible-only-bpe.tok < /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok/"+lang+filename_lang+" > /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok_bpe/"+lang+filename_lang[:-4]+".bpe.txt")
  call("python /home/cmalaviy/multi-nmt/subword-nmt/apply_bpe.py -c /projects/tir2/users/cmalaviy/bible-corpus/combined-bible-only-bpe.tok < /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok/"+lang+filename_lang+" > /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok_bpe/"+lang+filename_lang[:-4]+".bpe.txt", shell=True)
  call("python /home/cmalaviy/multi-nmt/subword-nmt/apply_bpe.py -c /projects/tir2/users/cmalaviy/bible-corpus/combined-bible-only-bpe.tok < /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok/"+lang+filename_en+" > /projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only_tok_bpe/"+lang+filename_en[:-4]+".bpe.txt", shell=True)

