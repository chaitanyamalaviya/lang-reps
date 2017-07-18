src_filename = "train.de-en-tok.de"
tgt_filename = "train.de-en-tok.en"
clean_src_filename = "clean.train.de-en-tok.de"
clean_tgt_filename = "clean.train.de-en-tok.en"

delete_lines = []

with open(src_filename) as f1:
  lines = f1.readlines()
  for i, line in enumerate(lines):
    if line.strip() and line[0]!='<':
      continue
    else:
      delete_lines.append(i)

with open(tgt_filename) as f1:
  lines = f1.readlines()
  for i, line in enumerate(lines):
    if line.strip() and line[0]!='<':
      continue
    elif i not in delete_lines:
      delete_lines.append(i)

with open(src_filename) as f1:
  with open(clean_src_filename,'w') as f2:
    lines = f1.readlines()
    for i, line in enumerate(lines):
      if i not in delete_lines:
        f2.write(line.lower())

with open(tgt_filename) as f1:
  with open(clean_tgt_filename,'w') as f2:
    lines = f1.readlines()
    for i, line in enumerate(lines):
      if i not in delete_lines:
        f2.write(line.lower())
