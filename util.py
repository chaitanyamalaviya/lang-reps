# coding: utf-8
from __future__ import absolute_import, print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math, ast, os, codecs, string
from subprocess import call
import cPickle as pickle
import sys, io, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import seaborn as sns
from pandas import DataFrame
#from matplotlib import rc
from sklearn.decomposition import PCA

# flatten = lambda l:[item for sublist in l for item in sublist]
# recursive_flatten = lambda l:flatten([recursive_flatten(item) if isinstance(item, list) else [item] for item in l])

def normalize(x):
    denom = sum(x)
    return [i/denom for i in x]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def sortbylength(src, tgt, maxlen):
    """
    :param src: List of source sentences in format [word1, word2..]
    :param tgt: List of target sentences in format [word1, word2..]
    :return: Sorted src and tgt lists
    """
    indexed_src = [(i,src[i]) for i in range(len(src))]
    sorted_indexed_src = sorted(indexed_src, key=lambda x: -len(x[1]))
    sorted_src = [item[1] for item in sorted_indexed_src if len(item[1])<maxlen]
    sort_order = [item[0] for item in sorted_indexed_src if len(item[1])<maxlen]
    sorted_tgt = [tgt[i] for i in sort_order]
    return sorted_src, sorted_tgt

def heatmap(src_sent, tgt_sent, att_weights, idx):

    plt.figure(figsize=(8, 6), dpi=80)
    att_probs = np.stack(att_weights, axis=1)
    
    plt.imshow(att_weights, cmap='gray', interpolation='nearest')
    #src_sent = [ str(s) for s in src_sent]
    #tgt_sent = [ str(s) for s in tgt_sent]
    #plt.xticks(range(0, len(tgt_sent)), tgt_sent, rotation='vertical')
    #plt.yticks(range(0, len(src_sent)), src_sent)
    plt.xticks(range(0, len(tgt_sent)),"")
    plt.yticks(range(0, len(src_sent)),"")
    plt.axis('off')
    plt.savefig("att_matrix_"+str(idx), bbox_inches='tight')
    plt.close()


def plot_trajectories(src_sent, src_encoding, idx):
    
    # encoding is (time_steps, hidden_dim)
    #pca = PCA(n_components=1)
    
    #pca_result = pca.fit_transform(src_encoding)
    times = np.arange(src_encoding.shape[0])
    plt.plot(times, src_encoding)
    plt.title(" ".join(src_sent))
    plt.xlabel('timestep')
    plt.ylabel('trajectories')
    plt.savefig("misc_hidden_cell_trajectories_"+str(idx), bbox_inches="tight")
    plt.close()


def plot_sent_trajectories(sents, decode_plot):
   
    font = {'family' : 'normal',
            'size'   : 14}

    matplotlib.rc('font', **font) 
    i = 0    
    l = ["Portuguese","Catalan"]
    
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([-1,1])

    for sent, enc in zip(sents, decode_plot):
	if i==2: continue
        i += 1
        #times = np.arange(len(enc))
        times = np.linspace(0,1,len(enc))
    	plt.plot(times, enc, label=l[i-1])
    plt.title("Hidden Node Trajectories")
    plt.xlabel('timestep')
    plt.ylabel('trajectories')
    plt.legend(loc='best')
    plt.savefig("final_tests/cr_por_cat_hidden_cell_trajectories", bbox_inches="tight")
    plt.close()

def itersubclasses(cls, _seen=None):
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

class Token(object):
    def __init__(self, i, s, count=1):
        self.i = i
        self.s = s
        self.count = count

    def __eq__(self, other):
        return self.i == other or self.s == other or \
               (isinstance(other, Token) and self.i == other.i and self.s == other.s)

    def __str__(self): return unicode(self.s)
    def __repr__(self): return str((self.s, self.i))
    def __hash__(self): return self.i

    @staticmethod
    def not_found(): raise Exception("token not found")



class Vocab(object):
    def __init__(self):
        self.tokens = set([])
        self.strings = set([])
        self.s2t = defaultdict(Token.not_found)
        self.i2t = defaultdict(Token.not_found)
        self.unk = None
        self.START_TOK = None
        self.END_TOK = None

    @property
    def size(self):
        return len(self.strings)

    def add(self, thing):
        if isinstance(thing, Token): self.add_token(thing)
        else: self.add_string(thing)

    def add_string(self, string):
        if string in self.strings:
            self[string].count += 1
            return self[string]
        i = len(self.tokens)
        s = string
        t = Token(i, s)
        self.i2t[i] = t
        self.s2t[s] = t
        self.tokens.add(t)
        self.strings.add(s)
        return t

    def add_token(self, tok):
        self.i2t[tok.i] = tok
        self.s2t[tok.s] = tok
        self.tokens.add(tok)
        self.strings.add(tok.s)
        return tok

    def __getitem__(self, key):
        if isinstance(key, int): return self.i2t[key]
        elif isinstance(key, Token): return key
        else: return self.s2t[key]

    def add_unk(self, thresh=0, unk_string='<UNK>'):
        if unk_string in self.s2t.keys(): raise Exception("tried to add an UNK token that already existed")
        if self.unk is not None: raise Exception("already added an UNK token")
        strings = [unk_string]
        for token in self.tokens:
            if token.count >= thresh: strings.append(token.s)
        if self.START_TOK is not None and self.START_TOK not in strings: strings.append(self.START_TOK.s)
        if self.END_TOK is not None and self.END_TOK not in strings: strings.append(self.END_TOK.s)
        self.tokens = set([])
        self.strings = set([])
        self.i2t = defaultdict(lambda :self.unk)
        self.s2t = defaultdict(lambda :self.unk)
        for string in strings:
            self.add_string(string)
        self.unk = self.s2t[unk_string]
        if self.START_TOK is not None: self.START_TOK = self.s2t[self.START_TOK.s]
        if self.END_TOK is not None: self.END_TOK = self.s2t[self.END_TOK.s]

    def pp(self, seq, delimiter=u''):
        return delimiter.join([unicode(self[item].s) for item in seq])

    def hpp(self, seq, delimiter=''):
        if isinstance(seq, int): return self.i2t[seq]
        else: return "["+delimiter.join([self.hpp(thing) for thing in seq])+"]"

    def save(self, filename):
        info_dict = {
            "tokens":self.tokens,
            "strings":self.strings,
            "s2t":dict(self.s2t),
            "i2t":dict(self.i2t),
            "unk":self.unk,
            "START_TOK":self.START_TOK,
            "END_TOK":self.END_TOK
        }
        with open(filename, "w") as f: pickle.dump(info_dict, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            info_dict = pickle.load(f)
            v = Vocab()
            v.tokens = info_dict["tokens"]
            v.strings = info_dict["strings"]
            v.unk = info_dict["unk"]
            v.START_TOK = info_dict["START_TOK"]
            v.END_TOK = info_dict["END_TOK"]
            defaultf = (lambda :v.unk) if (v.unk is not None) else Token.not_found
            v.s2t = defaultdict(defaultf, info_dict["s2t"])
            v.i2t = defaultdict(defaultf, info_dict["i2t"])
            return v

    @classmethod
    def load_from_corpus(cls, reader, remake=False, src_or_tgt="src"):
        vocab_fname = reader.fname+".vocab-"+reader.mode+"-"+src_or_tgt
        if not remake and os.path.isfile(vocab_fname):
            return Vocab.load(vocab_fname)
        else:
            v = Vocab()
            count = 0  # count of sentences
            for item in reader:
                toklist = item
                for token in toklist:
                    v.add(token)
                count += 1
                if count % 10000 == 0:
                    print("...", count, end="")
            print("\nSaving " + src_or_tgt + " vocab of size", v.size)
            v.START_TOK = v[reader.begin] if reader.begin is not None else None
            v.END_TOK = v[reader.end] if reader.end is not None else None
            v.save(vocab_fname)
            return v


#### reader class

class CorpusReaderTemplate(object):
    names = {"template",}

def get_reader(name):
    for c in itersubclasses(CorpusReaderTemplate):
        if name in c.names: return c
    raise Exception("no reader found with name: " + name)

class ParallelTranslationCorpus(CorpusReaderTemplate):
    names = {"parallel"}
    def __init__(self, fname, begin=None, end=None, mode="parallel"):
        self.fname = fname
        self.mode = mode
        self.begin = begin
        self.end = end
        self.seq2seq = True

    def __iter__(self):
        """
        Read a file where each line is of the form "word1 word2 ..."
        Yields lists of the form [word1, word2, ...]
        """
        if os.path.isdir(self.fname):
            filenames = [os.path.join(self.fname,f) for f in os.listdir(self.fname)]
        else:
            filenames = [self.fname]
        for filename in filenames:
            # with io.open(filename, encoding='utf-8') as f:
            with open(filename) as f:
                doc = f.read()
                for line in doc.split("\n"):
                    #if not line:  continue
                    sent = "".join([ch for ch in line.lower() if ch not in string.punctuation]).strip().split()
                    # sent = [word for word in line.strip().split()]
                    sent = [self.begin] + sent + [self.end]
                    yield sent


class BibleTranslationCorpus(CorpusReaderTemplate):

    "Add language vector (Diff for each language, maybe dict of vectors) + read data from bible corpus"

    names = {"bible"}
    def __init__(self, fname, begin=None, end=None, mode="bible"):
        self.fname = fname
        self.mode = mode
        self.begin = begin
        self.end = end
        self.seq2seq = True
        self.lang = None

    def __iter__(self):
        """
        Read a file where each line is of the form "word1 word2 ..."
        Yields lists of the form [word1, word2, ...]
        """
        #jfbbb
	if os.path.isdir(self.fname):
            filenames = [os.path.join(self.fname,f) for f in os.listdir(self.fname)]
        #else:
        #    filenames = [self.fname]
        
        for langpath in filenames:
            with open(filename) as f:
                doc = f.read()
                for line in doc.split("\n"):
                    #if not line:  continue
                    sent = "".join([ch for ch in line.lower() if ch not in string.punctuation]).strip().split()
                    # sent = [word for word in line.strip().split()]
                    sent = [self.begin] + sent + [self.end]
                    yield sent
