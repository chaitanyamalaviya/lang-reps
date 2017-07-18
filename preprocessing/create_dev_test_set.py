import random
val_size = test_size = 3000

### SOURCE

with open("/projects/tir2/users/cmalaviy/bible-corpus/comb_src_lang.txt") as f:
  doc = f.read()
  doc_tok = [line for line in doc.split("\n")]

maxlen = len(doc_tok)
val_idx = random.sample(range(0, maxlen), val_size+test_size)
test_idx = val_idx[:test_size]
val_idx = val_idx[test_size:]

val_set = []
test_set = []
train_set = []

for idx in val_idx:
  val_set.append(doc_tok[idx])
for idx in test_idx:
  test_set.append(doc_tok[idx])

for idx in range(maxlen):
  if idx not in val_idx and idx not in test_idx:
    train_set.append(doc_tok[idx])
    
with open('/projects/tir2/users/cmalaviy/bible-corpus/train_src_lang.txt','a') as f:
  f.write("\n".join(train_set))

with open('/projects/tir2/users/cmalaviy/bible-corpus/val_src_lang.txt','a') as f:
  f.write("\n".join(val_set))

with open('/projects/tir2/users/cmalaviy/bible-corpus/test_src_lang.txt','a') as f:
  f.write("\n".join(test_set))

### TARGET

with open("/projects/tir2/users/cmalaviy/bible-corpus/comb_tgt_en.txt") as f:
  doc = f.read()
  doc_tok = [line for line in doc.split("\n")]

train_set = []
val_set = []
test_set = []

for idx in val_idx:
  val_set.append(doc_tok[idx])
for idx in test_idx:
  test_set.append(doc_tok[idx])

for idx in range(maxlen):
  if idx not in val_idx and idx not in test_idx:
    train_set.append(doc_tok[idx])

with open('/projects/tir2/users/cmalaviy/bible-corpus/train_tgt_en.txt','a') as f:
  f.write("\n".join(train_set))

with open('/projects/tir2/users/cmalaviy/bible-corpus/val_tgt_en.txt','a') as f:
  f.write("\n".join(val_set))

with open('/projects/tir2/users/cmalaviy/bible-corpus/test_tgt_en.txt','a') as f:
  f.write("\n".join(test_set))
