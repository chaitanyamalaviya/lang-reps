# coding: utf-8
from  __future__ import absolute_import, print_function, unicode_literals, division
import dynet
import random, util, sys, math, time
import argparse
import numpy as np
import sequence2sequence as s2s
import embeddings

np.random.seed(442551802)
random.seed(442551802)

# input args
parser = argparse.ArgumentParser()

## dummy argument for dynet
parser.add_argument("--dynet-mem")

## locations of data
# a single file or directory for each argument
parser.add_argument("--train_src")
parser.add_argument("--train_tgt")
parser.add_argument("--valid_src")
parser.add_argument("--valid_tgt")
parser.add_argument("--test_src")
parser.add_argument("--test_tgt")
parser.add_argument("--reader_mode", default="parallel") # format of corpus (parallel..etc)

## alternatively, load one dataset and split it
parser.add_argument("--percent_valid", default=3000, type=int)

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=2, type=int)

## rnn parameters
parser.add_argument("--layers", default=1, type=int)
parser.add_argument("--input_dim", default=512, type=int)
parser.add_argument("--hidden_dim", default=512, type=int)
parser.add_argument("--attention_dim", default=256, type=int)
parser.add_argument("--rnn", default="lstm")
parser.add_argument("--trainer", default="adam")  # from (simple_sgd, momentum_sgd, adadelta, adagrad, adam)

## word embedding specific parameters
# loss function to be used among/or combination of: phonological/morphological, bilingual, regularization, external knowledge
parser.add_argument("--loss_function", default="cross_entropy")
parser.add_argument("--write_embeddings", action='store_true') # Write embeddings to file
parser.add_argument("--extract_lvs", type=str)

## experiment parameters
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--log_train_every_n", default=5000, type=int)
parser.add_argument("--log_valid_every_n", default=40000, type=int)
parser.add_argument("--log_output")

## choose what model to use
parser.add_argument("--model_type", default="basic")
parser.add_argument("--load")
parser.add_argument("--save")
parser.add_argument("--eval", action='store_true')
parser.add_argument("--eval_all", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--results_filename", default='results')

## model-specific parameters
parser.add_argument("--beam_size", default=3, type=int)
parser.add_argument("--minibatch_size", default=64, type=int)

args = parser.parse_args()
print("ARGS:", args)

if args.rnn == "lstm": args.rnn = dynet.LSTMBuilder
elif args.rnn == "gru": args.rnn = dynet.GRUBuilder
else: args.rnn = dynet.SimpleRNNBuilder

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'

# define model and obtain vocabulary
# (reload vocab files if saved model or create new vocab files if new model)

model = dynet.Model()
if not args.trainer or args.trainer=="simple_sgd":
    trainer = dynet.SimpleSGDTrainer(model)
elif args.trainer == "momentum_sgd":
    trainer = dynet.MomentumSGDTrainer(model)
elif args.trainer == "adadelta":
    trainer = dynet.AdadeltaTrainer(model)
elif args.trainer == "adagrad":
    trainer = dynet.AdagradTrainer(model)
elif args.trainer == "adam":
    trainer = dynet.AdamTrainer(model)
else:
    raise Exception("Trainer not recognized! Please use one of {simple_sgd, momentum_sgd, adadelta, adagrad, adam}")

# Set sparse updates for efficiency
trainer.set_sparse_updates(True)

# Load train/valid corpus

print("Loading corpus...")
train_data_src = list(util.get_reader(args.reader_mode)(args.train_src, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
train_data_tgt = list(util.get_reader(args.reader_mode)(args.train_tgt, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
if args.valid_src and args.valid_tgt:
    valid_data_src = list(util.get_reader(args.reader_mode)(args.valid_src, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    valid_data_tgt = list(util.get_reader(args.reader_mode)(args.valid_tgt, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
else:
    if args.percent_valid > 1: cutoff = args.percent_valid
    else: cutoff = int(len(train_data_src)*(args.percent_valid))

    valid_data_src = train_data_src[-cutoff:]
    valid_data_tgt = train_data_tgt[-cutoff:]

    train_data_src = train_data_src[:-cutoff]
    train_data_tgt = train_data_tgt[:-cutoff]

print("Train set of size", len(train_data_src), "/ Validation set of size", len(valid_data_src))
assert len(train_data_src)==len(train_data_tgt), "Number of source and target sentences in training set not equal!"
assert len(valid_data_src)==len(valid_data_tgt), "Number of source and target sentences in validation set not equal!"
print("done.")

## Sort train/valid set before minibatching
train_data_src, train_data_tgt = util.sortbylength(train_data_src, train_data_tgt, 80)
valid_data_src, valid_data_tgt = util.sortbylength(valid_data_src, valid_data_tgt, 80)

# Initialize model
S2SModel = s2s.get_s2s(args.model_type)
if args.load:
    print("Loading existing model...")
    s2s = S2SModel.load(model, train_data_src, train_data_tgt, args.load)
    src_vocab = s2s.src_vocab
    tgt_vocab = s2s.tgt_vocab
else:
    print("New model. Getting vocabulary from training set...")

    src_reader = util.get_reader(args.reader_mode)(args.train_src, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
    src_vocab = util.Vocab.load_from_corpus(src_reader, remake=args.rebuild_vocab, src_or_tgt="src")
    src_vocab.START_TOK = src_vocab[BEGIN_TOKEN]
    src_vocab.END_TOK = src_vocab[END_TOKEN]
    src_vocab.add_unk(args.unk_thresh)

    tgt_reader = util.get_reader(args.reader_mode)(args.train_tgt, mode=args.reader_mode, end=END_TOKEN)
    tgt_vocab = util.Vocab.load_from_corpus(tgt_reader, remake=args.rebuild_vocab, src_or_tgt="tgt")
    tgt_vocab.END_TOK = tgt_vocab[END_TOKEN]
    tgt_vocab.add_unk(args.unk_thresh)

    print("Source vocabulary of size", src_vocab.size, "and target vocab of size", tgt_vocab.size)
    print("Creating model...")
    s2s = S2SModel(model, src_vocab, tgt_vocab, args)
    print("...done!")


if args.extract_lvs:
    print("Writing language vectors...")
    embeddings.write_embeddings(s2s.src_lookup, src_vocab, args.extract_lvs)
    sys.exit("...done.")

# create log file for training
if args.log_output:
    outfile = open(args.log_output, 'w')
    outfile.write("")
    outfile.close()

if args.eval_all:
    print("Evaluating for all languages..")
    s2s.all_langs_translate()

# only evaluate existing model on test data
if args.eval:
    print("Evaluating model...")
    test_data_src = list(util.get_reader(args.reader_mode)(args.test_src, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    test_data_tgt = list(util.get_reader(args.reader_mode)(args.test_tgt, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    with open(args.test_src) as f:
	lines = f.readlines()
    #combined = list(zip(test_data_src, test_data_tgt))
    #random.shuffle(combined)
    #test_data_src[:], test_data_tgt[:] = zip(*combined)
    test_data = zip(test_data_src, test_data_tgt)
    s2s.translate(lines, test_data, args.results_filename, "final")
    if args.test:
        s2s.evaluate(test_data)
        sys.exit("...done.")
    else: raise Exception("Test file path argument missing")
    sys.exit("...done.")
    if args.plot_embeddings:
        s2s.tsne_embeddings(test_data)
        sys.exit("...done.")

# Shuffle training data for bible model
#combined = list(zip(train_data_src, train_data_tgt))
#random.shuffle(combined)
#train_data_src[:], train_data_tgt[:] = zip(*combined)

# Store starting index of each minibatch
if args.minibatch_size != 0:
    train_order = [x*args.minibatch_size for x in range(int(len(train_data_src)/args.minibatch_size + (1 if len(train_data_src)%args.minibatch_size != 0 else 0)))]
    valid_order = [x*args.minibatch_size for x in range(int(len(valid_data_src)/args.minibatch_size + (1 if len(valid_data_src)%args.minibatch_size != 0 else 0)))]
else:
    train_order = range(len(train_data))
    valid_order = range(len(valid_data))

# run training loop
word_count = sent_count = cum_loss = 0.0
try:
    for ITER in range(args.epochs):
        s2s.epoch = ITER
        random.shuffle(train_order)
        sample_num = 0
        _start = time.time()
        log_start = time.time()
        for i, sid in enumerate(train_order):
            # Retrieving batch from training data
            batched_src = train_data_src[sid : sid + args.minibatch_size]
            batched_tgt = train_data_tgt[sid : sid + args.minibatch_size]

            sample_num = 1 + i

            if sample_num % (int(args.log_train_every_n/args.minibatch_size)) == 0:
                print("[training_set] Epoch:", ITER, "Batch:", sample_num)
                trainer.status()
                print("Loss:", cum_loss / word_count, "Time elapsed:", (time.time() - _start) ,"WPS:", word_count/(time.time() - log_start))
                # sample = lm.beam_search_generate(src, beam_n=args.beam_size)
                # sample = s2s.generate(src, sampled=False)
                word_count = sent_count = cum_loss = 0.0
                log_start = time.time()
                print
            # end of test logging

            if sample_num % (int(args.log_valid_every_n/args.minibatch_size)) == 0:
                v_word_count = v_sent_count = v_cum_loss = v_cum_bleu = v_cum_em = 0.0
                v_start = time.time()
                for vid in valid_order:
                    batched_v_src = valid_data_src[vid : vid + args.minibatch_size]
                    batched_v_tgt = valid_data_tgt[vid : vid + args.minibatch_size]
                    v_loss = s2s.get_batch_loss(batched_v_src, batched_v_tgt)
                    v_cum_loss += v_loss.scalar_value()
                    # v_cum_em += s2s.get_em(batched_v_src, batched_v_tgt)
                    # v_cum_bleu += s2s.get_bleu(v_src, v_tgt, args.beam_size)
                    v_word_count += sum([(len(tgt_sent) - 1) for tgt_sent in batched_v_tgt])
                    v_sent_count += args.minibatch_size
                print("[Validation Set", str(sample_num) + "]\t" + \
                      "Loss:", str(v_cum_loss / v_word_count) + "\t" + \
                      "Perplexity:", str(np.exp(v_cum_loss / v_word_count)) + "\t" + \
                      # "BLEU: "+str(v_cum_bleu / v_sent_count) + "\t" + \
                      # "EM: "  +str(v_cum_em   / v_sent_count) + "\t" + \
                      "Time elapsed:", str(time.time() - v_start))
                if args.log_output:
                    print("(logging to", args.log_output + ")")
                    with open(args.log_output, "a") as outfile:
                        outfile.write(str(ITER) + "\t" + \
                                      str(sample_num) + "\t" + \
                                      str(v_cum_loss / v_word_count) + "\t" + \
                                      str(np.exp(v_cum_loss / v_word_count)) + "\n")
                                      # str(v_cum_em   / v_sent_count) + "\t" + \
                                      # str(v_cum_bleu / v_sent_count) + "\n")
                
                s2s.translate(zip(valid_data_src, valid_data_tgt), args.results_filename, sample_num, ITER)
                
                if args.save:
                  print("saving checkpoint...")
                  s2s.save(args.save + ".checkpoint")
            # end of validation logging
            loss = s2s.get_batch_loss(batched_src, batched_tgt)
            loss_value = loss.value()
            cum_loss += loss_value * args.minibatch_size
            word_count += sum([(len(tgt_sent) - 1) for tgt_sent in batched_tgt])
            sent_count += args.minibatch_size

            ppl = np.exp((loss_value * args.minibatch_size) / word_count)

            loss.backward()
            trainer.update()

            ### end of batch train loop
        trainer.update_epoch()
        ### end of epoch
    ### end of training loop

except KeyboardInterrupt:
    if args.save:
        print("saving...")
        model.save(args.save)
    print("Unexpected error:", sys.exc_info()[0])
    raise

if args.save:
    print("saving...")
    s2s.save(args.save)
