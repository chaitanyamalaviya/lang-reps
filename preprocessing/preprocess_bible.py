import os
import glob

fname = "/projects/tir2/users/cmalaviy/bible-corpus/bible-corpus/"
output_dir = "/projects/tir2/users/cmalaviy/bible-corpus/combined_bible_only/"
language_set = [f for f in os.listdir(fname)]
src_paths = [glob.glob(fname + l + "/parallel_text/bible_com*/" + l) for l in language_set]

def write_combined_file(lang_code, all_lang_paths, all_en_paths):
  src_combined_filename = "train" + "_" + lang_code + "_en."+ lang_code + ".txt"
  tgt_combined_filename = "train" + "_" + lang_code + "_en.en" + ".txt"
  
  if not os.path.exists(output_dir + lang_code):
    os.makedirs(output_dir+lang_code)

  write_lang = []
  write_en = []

  for corp in all_lang_paths:
    for filename in corp:
      with open(filename) as f:
        doc = f.read()
        write_lang.append(doc)

  for corp in all_en_paths:
    for filename in corp:
      with open(filename) as f:
	doc = f.read()
        write_en.append(doc)

  for doc1, doc2 in zip(write_lang, write_en):
    if len(doc1.split("\n"))!=len(doc2.split("\n")):
      continue
    else:
      with open(output_dir + lang_code + "/" + src_combined_filename, 'a') as wf:
        wf.write(doc1)
      with open(output_dir + lang_code + "/" + tgt_combined_filename, 'a') as wf:
        wf.write(doc2)

for lang_path in src_paths:
  
  all_lang_paths = []
  all_en_paths = []
  if lang_path:
    lang_code = lang_path[0][-3:].lower()
  
    for f in lang_path:
   
      l_files = [os.path.join(f,p) for p in os.listdir(f)]
      en_files = [p[:70] + p[70:].replace("/"+lang_code+"/", '/eng/') for p in l_files]
      #en_files = [os.path.join(f[:-3]+"eng/",p) for p in os.listdir(f[:-3]+"eng/")]
      all_lang_paths.append(l_files)
      all_en_paths.append(en_files)
    
     #print(all_lang_paths, all_en_paths)
    if len(all_lang_paths)==len(all_en_paths): write_combined_file(lang_code, all_lang_paths, all_en_paths)
    else: print("Skipping lang",lang_path[:-3])
