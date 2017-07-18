#encoding: :utf-8
from  __future__ import absolute_import, print_function, unicode_literals
from itertools import izip
from nltk.translate.bleu_score import corpus_bleu
from kitchen.text.converters import getwriter
import dynet
import random, math, os, util
import numpy as np
import cPickle as pickle
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
UTF8Writer = getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

class Seq2SeqTemplate(object):
    name = "template"

def get_s2s(name):
    for c in util.itersubclasses(Seq2SeqTemplate):
        if c.name == name: return c
    raise Exception("No seq2seq model found with name: " + name)

class Seq2SeqBasic(Seq2SeqTemplate):
    """
    Bidirectional LSTM encoder and unidirectional decoder without attention
    """
    name = "basic"

    def __init__(self, model, train_data_src, train_data_tgt, src_vocab, tgt_vocab, args):

        self.m = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args

        # Bidirectional Encoder LSTM
        print("Adding Forward encoder LSTM parameters")
        self.enc_fwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)
        print("Adding Backward encoder LSTM parameters")
        self.enc_bwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)

        #Decoder LSTM
        print("Adding decoder LSTM parameters")
        self.dec_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)

        #Decoder weight and bias
        print("Adding Decoder weight")
        self.decoder_w = model.add_parameters( (tgt_vocab.size, args.hidden_dim))
        print("Adding Decoder bias")
        self.decoder_b = model.add_parameters( (tgt_vocab.size,))

        #Lookup parameters
        print("Adding lookup parameters")
        self.src_lookup = model.add_lookup_parameters( (src_vocab.size, args.input_dim))
        self.tgt_lookup = model.add_lookup_parameters( (tgt_vocab.size, args.input_dim))

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.src_vocab.save(path+"/vocab.src")
        self.tgt_vocab.save(path+"/vocab.tgt")
        self.m.save(path+"/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @classmethod
    def load(cls, model, train_data_src, train_data_tgt, path):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        src_vocab = util.Vocab.load(path+"/vocab.src")
        tgt_vocab = util.Vocab.load(path+"/vocab.tgt")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        s2s = cls(model, src_vocab, tgt_vocab, args)
        s2s.m.load(path+"/params")
        return s2s

    def embed_seq(self, seq):
        """
        Embedding for a single sentence
        :param seq: sentence
        :return: Word embeddings
        """
        wembs = [self.src_lookup[self.src_vocab[tok].i] for tok in seq]
        return wembs

    def embed_batch_seq(self, wids):
        """
        Embedding method for a batch of sentences
        :param wids: Word IDs for a batch of sentences
        :return: Word embedding matrix
        """

        wembs_batch = [dynet.lookup_batch(self.src_lookup, wid) for wid in wids]
        return wembs_batch

    def encode_seq(self, src_seq):
        """
        Encode a single sentence
        :param src_seq: source sentence
        :return: encoded vector
        """

        src_seq_rev = list(reversed(src_seq))
        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = self.enc_bwd_lstm.initial_state().transduce(src_seq_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dynet.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def encode_batch_seq(self, src_seq, src_seq_rev):
        """
        Encodes a batch of sentences
        :param src_seq: batch of sentences
        :param src_seq_rev: batch of sentences in reversed order
        :return: last hidden state of the encoder
        """
        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = list(reversed(self.enc_bwd_lstm.initial_state().transduce(src_seq_rev)))
        return dynet.concatenate([fwd_vectors[-1], bwd_vectors[-1]])

    def decode(self, encoding, input, output):
        """
        Single training example decoding function
        :param encoding: last hidden state from encoder
        :param input: source sentence
        :param output: target sentence
        :return: loss value
        """

        src_toks = [self.src_vocab[tok] for tok in input]
        tgt_toks = [self.tgt_vocab[tok] for tok in output]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        s = self.dec_lstm.initial_state().add_input(encoding)
        loss = []

        sent = []
        for tok in tgt_toks:
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            cross_ent_loss = - dynet.log(dynet.pick(probs, tok.i))
            loss.append(cross_ent_loss)
            embed_vector = self.tgt_lookup[tok.i]
            s = s.add_input(embed_vector)

        loss = dynet.esum(loss)
        return loss


    def decode_batch(self, encoding, output_batch):

        """
        Batch decoding function
        :param encoding: last hidden state from encoder
        :param output_batch: list of output sentences in format [word1, word2..]
        :return: loss
        """
        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        s = self.dec_lstm.initial_state().add_input(encoding)
        losses = []

        maxSentLength = max([len(sent) for sent in output_batch])
        wids = []
        masks = []
        for j in range(maxSentLength):
            wids.append([(self.tgt_vocab[sent[j]].i if len(sent)>j else self.tgt_vocab.END_TOK.i) for sent in output_batch])
            mask = [(1 if len(sent)>j else 0) for sent in output_batch]
            masks.append(mask)

        for wid, mask in zip(wids, masks):

            # apply dropout
            y = s.output()
            if args.dropout: y = dynet.dropout(y, self.args.dropout)

            # calculate the softmax and loss
            score = dynet.affine_transform([b, w, y])
            loss = dynet.pickneglogsoftmax_batch(score, wid)

            # mask the loss if at least one sentence is shorter than maxSentLength
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), len(mask))
                loss = loss * mask_expr

            losses.append(loss)

            # update the state of the RNN
            embed_vector = dynet.lookup_batch(self.tgt_lookup, wid)
            s = s.add_input(embed_vector)

        return dynet.sum_batches(dynet.esum(losses))


    def generate(self, src, sampled=False):
        dynet.renew_cg()

        embedding = self.embed_seq(src)
        encoding = self.encode_seq(embedding)[-1]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state().add_input(encoding)

        out = []
        for _ in range(5*len(src)):
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            selection = np.argmax(probs.value())
            out.append(self.tgt_vocab[selection])
            if out[-1].s == self.tgt_vocab.END_TOK: break
            embed_vector = self.tgt_lookup[selection]
            s = s.add_input(embed_vector)
        return out

    def beam_search_generate(self, src_seq, beam_n=5):
        dynet.renew_cg()

        embedded = self.embed_seq(src_seq)
        input_vectors = self.encode_seq(embedded)

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(input_vectors[-1])
        beams = [{"state":  s,
                  "out":    [],
                  "err":    0}]
        completed_beams = []
        while len(completed_beams) < beam_n:
            potential_beams = []
            for beam in beams:
                if len(beam["out"]) > 0:
                    embed_vector = self.tgt_lookup[beam["out"][-1].i]
                    s = beam["state"].add_input(embed_vector)

                out_vector = dynet.affine_transform([b, w, s.output()])
                probs = dynet.softmax(out_vector)
                probs = probs.vec_value()

                for potential_next_i in range(len(probs)):
                    potential_beams.append({"state":    s,
                                            "out":      beam["out"]+[self.tgt_vocab[potential_next_i]],
                                            "err":      beam["err"]-math.log(probs[potential_next_i])})

            potential_beams.sort(key=lambda x:x["err"])
            beams = potential_beams[:beam_n-len(completed_beams)]
            completed_beams = completed_beams+[beam for beam in beams if beam["out"][-1] == self.tgt_vocab.END_TOK
                                                                      or len(beam["out"]) > 5*len(src_seq)]
            beams = [beam for beam in beams if beam["out"][-1] != self.tgt_vocab.END_TOK
                                            and len(beam["out"]) <= 5*len(src_seq)]
        completed_beams.sort(key=lambda x:x["err"])
        return [beam["out"] for beam in completed_beams]


    def get_loss(self, input, output):

        dynet.renew_cg()

        embedded = self.embed_seq(input)
        encoded = self.encode_seq(embedded)[-1]
        return self.decode(encoded, input, output)


    def get_batch_loss(self, input_batch, output_batch):

        dynet.renew_cg()

        # Dimension: maxSentLength * minibatch_size
        wids = []
        wids_reversed = []

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        # masks = [] # Dimension: maxSentLength * minibatch_size

        # tot_words = 0
        maxSentLength = max([len(sent) for sent in input_batch])
        for j in range(maxSentLength):
            wids.append([(self.src_vocab[sent[j]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            wids_reversed.append([(self.src_vocab[sent[len(sent)- j-1]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            # mask = [(1 if len(sent)>j else 0) for sent in input_batch]
            # masks.append(mask)
            #tot_words += sum(mask)

        embedded_batch = self.embed_batch_seq(wids)
        embedded_batch_reverse = self.embed_batch_seq(wids_reversed)
        encoded_batch = self.encode_batch_seq(embedded_batch, embedded_batch_reverse)

        # pass last hidden state of encoder to decoder
        return self.decode_batch(encoded_batch, output_batch)


    def get_bleu(self, input, output, beam_n=5):

        guess = self.generate(input, sampled=False)
        input_str = [tok.s for tok in guess]
        output_str = [tok.s for tok in output]
        ans = BLEU.sentence_bleu(input_str, output_str)
        return ans


    def get_em(self, input, output, beam_n=5):

        guess = self.generate(input, sampled=False)
        input_str = [tok.s for tok in guess]
        output_str = [tok.s for tok in output]
        ans = 1 if input_str == output_str else 0
        return ans


class Hypothesis(object):
    def __init__(self, state, y, ctx_tm1, score, alpha):
        self.state = state
        self.y = y
        self.ctx_tm1 = ctx_tm1
        self.score = score
        self.alpha = alpha
     
class Seq2SeqBiRNNAttn(Seq2SeqBasic):
    """
    Bidirectional LSTM encoder and unidirectional decoder with attention
    """
    name = "attention"

    def __init__(self, model, src_vocab, tgt_vocab, args):
        self.m = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args
        # Bidirectional Encoder LSTM
        print("Adding Forward encoder LSTM parameters")
        self.enc_fwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)
        print("Adding Backward encoder LSTM parameters")
        self.enc_bwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)

        #Decoder LSTM
        print("Adding decoder LSTM parameters")
        self.dec_lstm = dynet.LSTMBuilder(args.layers, args.input_dim + args.hidden_dim*2, args.hidden_dim, model)

        if args.dropout > 0.:
            self.enc_fwd_lstm.set_dropout(args.dropout)
            self.enc_bwd_lstm.set_dropout(args.dropout)
            self.dec_lstm.set_dropout(args.dropout)

        #Decoder weight and bias
        print("Adding Decoder weight")
        self.decoder_w = model.add_parameters( (tgt_vocab.size, args.hidden_dim))
        print("Adding Decoder bias")
        self.decoder_b = model.add_parameters( (tgt_vocab.size,))
        self.decoder_b.zero()

        # transformation of decoder hidden states and context vectors before reading out target words
        self.W_h = model.add_parameters((args.input_dim, args.hidden_dim + args.hidden_dim * 2))
        self.b_h = model.add_parameters((args.input_dim))
        self.b_h.zero()

        # transformation of context vectors at t_0 in decoding
        self.W_s = model.add_parameters((args.hidden_dim, args.hidden_dim * 2))
        self.b_s = model.add_parameters((args.hidden_dim))
        self.b_s.zero()

        print("Adding lookup parameters")
        #Lookup parameters
        self.src_lookup = model.add_lookup_parameters( (src_vocab.size, args.input_dim))
        self.tgt_lookup = model.add_lookup_parameters( (tgt_vocab.size, args.input_dim))

        #Attention parameters
        print("Adding Attention Parameters")
        self.attention_w1 = model.add_parameters( (args.attention_dim, args.hidden_dim * 2))
        self.attention_w2 = model.add_parameters( (args.attention_dim, args.hidden_dim ))
        self.attention_v = model.add_parameters( (1, args.attention_dim))


    def embed_seq(self,wids):
        return [dynet.lookup(self.src_lookup, wid) for wid in wids]

    def encode_seq(self, src_seq, src_seq_rev):
        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = list(reversed(self.enc_fwd_lstm.initial_state().transduce(src_seq_rev)))
        return [dynet.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    def embed_batch_seq(self, wids):

        wembs_batch = [dynet.lookup_batch(self.src_lookup, wid) for wid in wids]
        return wembs_batch


    def encode_batch_seq(self, src_seq, src_seq_rev):

        forward_states = self.enc_fwd_lstm.initial_state().add_inputs(src_seq)
        backward_states = self.enc_bwd_lstm.initial_state().add_inputs(src_seq_rev)[::-1]

        src_encodings = []
        forward_cells = []
        backward_cells = []
        for forward_state, backward_state in zip(forward_states, backward_states):
            fwd_cell, fwd_enc = forward_state.s()
            bak_cell, bak_enc = backward_state.s()

            src_encodings.append(dynet.concatenate([fwd_enc, bak_enc]))
            forward_cells.append(fwd_cell)
            backward_cells.append(bak_cell)

        decoder_init = dynet.concatenate([forward_cells[-1], backward_cells[0]])
	decoder_all = [dynet.concatenate([fwd, bwd]) for fwd, bwd in zip(forward_cells, list(reversed(backward_cells)))]
        return src_encodings, decoder_all


    def attend(self, input_vectors, state, batch_size):

        w1 = dynet.parameter(self.attention_w1)
        w2 = dynet.parameter(self.attention_w2)
        v = dynet.parameter(self.attention_v)

        src_len = len(input_vectors)

        # enc_size, sent_len, batch_size
        src_enc_all = dynet.concatenate_cols(input_vectors)

        att_hidden = dynet.tanh(dynet.colwise_add(w1 * src_enc_all, w2 * state))
        att_weights = dynet.reshape(v * att_hidden, (src_len, ), batch_size)
        # sent_len, batch_size
        att_weights = dynet.softmax(att_weights)

        output_vectors = src_enc_all * att_weights

        return output_vectors, att_weights

    def decode(self, input_vectors, output):

        tgt_toks = [self.tgt_vocab[tok] for tok in output]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(dynet.concatenate([
                                            input_vectors[-1],
                                            dynet.vecInput(self.args.hidden_dim)
                                          ]))
        loss = []
        for tok in tgt_toks:
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            loss.append(-dynet.log(dynet.pick(probs, tok.i)))
            embed_vector = self.tgt_lookup[tok.i]
            attn_vector = self.attend(input_vectors, s)
            inp = dynet.concatenate([embed_vector, attn_vector])
            s = s.add_input(inp)

        loss = dynet.esum(loss)
        return loss


    def decode_batch(self, encoding, output_batch, decoder_init, input_batch):

        W_s = dynet.parameter(self.W_s)
        b_s = dynet.parameter(self.b_s)
        W_h = dynet.parameter(self.W_h)
        b_h = dynet.parameter(self.b_h)
        W_y = dynet.parameter(self.decoder_w)
        b_y = dynet.parameter(self.decoder_b)

        maxSentLength = max([len(sent) for sent in output_batch])
        wids = []
        masks = []
        for j in range(maxSentLength):
            wids.append([(self.tgt_vocab[sent[j]].i if len(sent)>j else self.tgt_vocab.END_TOK.i) for sent in output_batch])
            mask = [(1 if len(sent)>j else 0) for sent in output_batch]
            masks.append(mask)


        decoder_init_cell = dynet.affine_transform([b_s, W_s, decoder_init])
        s = self.dec_lstm.initial_state([decoder_init_cell, dynet.tanh(decoder_init_cell)])
        # s = self.dec_lstm.initial_state.add_input([decoder_init_cell, dynet.tanh(decoder_init_cell)])
        ctx_tm1 = dynet.vecInput(self.args.hidden_dim * 2)
        losses = []

        # start from <S>, until y_{T-1}
        for t, (y_ref_t, mask_t) in enumerate(zip(wids[1:], masks[1:]), start=1):
            y_tm1_embed = dynet.lookup_batch(self.tgt_lookup, wids[t - 1])
            x = dynet.concatenate([y_tm1_embed, ctx_tm1])
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attend(encoding, h_t, len(output_batch))

            # read_out = dynet.tanh(W_h * dynet.concatenate([h_t, ctx_t]) + b_h)
            read_out = dynet.tanh(dynet.affine_transform([b_h, W_h, dynet.concatenate([h_t, ctx_t])]))
            if self.args.dropout > 0.:
                read_out = dynet.dropout(read_out, self.args.dropout)
            # y_t = W_y * read_out + b_y
            y_t = dynet.affine_transform([b_y, W_y, read_out])
            loss_t = dynet.pickneglogsoftmax_batch(y_t, y_ref_t)

            if 0 in mask_t:
                mask_expr = dynet.inputVector(mask_t)
                mask_expr = dynet.reshape(mask_expr, (1, ), len(output_batch))
                loss_t = loss_t * mask_expr

            losses.append(loss_t)
            ctx_tm1 = ctx_t

        return dynet.sum_batches(dynet.esum(losses)) / len(output_batch)


    def get_batch_loss(self, input_batch, output_batch):

        dynet.renew_cg()

        # Dimension: maxSentLength * minibatch_size
        wids = []
        # wids_reversed = []

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        # masks = [] # Dimension: maxSentLength * minibatch_size

        maxSentLength = max([len(sent) for sent in input_batch])

        for j in range(maxSentLength):
            wids.append([(self.src_vocab[sent[j]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])

        embedded_batch = self.embed_batch_seq(wids)
        embedded_batch_reverse = embedded_batch[::-1]
        encoded_batch, decoder_init = self.encode_batch_seq(embedded_batch, embedded_batch_reverse)

        # pass all hidden states of encoder to decoder (for attention)
        return self.decode_batch(encoded_batch, output_batch, decoder_init, input_batch)


    def get_loss(self, input, output):

        dynet.renew_cg()

        embedded = self.embed_seq(input)
        encoded = self.encode_seq(embedded)
        return self.decode(encoded, output)


    def translate(self, lines, test_data, filename, val_idx, cell_idx=0, epoch=""):
        if not os.path.exists("final_tests"):
          os.makedirs("final_tests")
 
        translations = []
        references = []
        empty = True
        f = open("final_tests/" + filename + "_" + str(epoch) + "_" + str(val_idx) + ".txt", "a")
        idx = 0
        dec_plot = []
	sents = []
        for src_sent, tgt_sent in test_data:
            dynet.renew_cg()           
            wids = [self.src_vocab[tok].i for tok in src_sent]
            embedded_seq = self.embed_seq(wids)
            embedded_seq_rev = embedded_seq[::-1]
            src_encodings, decoder_init = self.encode_batch_seq(embedded_seq, embedded_seq_rev)
            
            h, src_encodings = self.beam_translate(src_sent, self.args.beam_size)
            h = h[0]
            alpha = h.alpha
            sample = h.y
            decoder_init = [enc.npvalue() for enc in decoder_init]
            dec_plot = [dec[cell_idx] for dec in decoder_init]
            src = [self.src_vocab[tok].s for tok in src_sent[1:]]
            tgt = [self.tgt_vocab[tok].s for tok in tgt_sent]
            hyp = [self.tgt_vocab[tok].s for tok in sample]

            if len(hyp) > 3:
                empty = False
                references.append([tgt])
                translations.append(hyp)
                f.writelines(["***************************\n",
                   "Source Sent: ", u" ".join(src[1:-1])+"\n",
                   "Target Sent: ", u" ".join(tgt[1:-1])+"\n",
                   "Generated: ", u" ".join(hyp[1:-1])+"\n"])
                
            last_encs[idx] = src_encodings[-1]
            #util.heatmap(src, tgt, alpha, idx)
            sents.append(src)
            dec_plot.append([dec[cell_idx] for dec in decoder_init])
            #util.plot_trajectories(sent, np.asarray(dec_plot), idx)
            #util.plot_nodes(src, src_encodings, idx)
            idx += 1
        util.plot_sent_trajectories(sents, dec_plot) 
        if empty:
            return 0.0, translations
        #mean = np.mean(last_encs, axis=0)
        #var = np.var(last_encs, axis=0)
        if len(translations)==0: bleu_score=0.0
        else: bleu_score = corpus_bleu(references, translations)
        f.write("BLEU SCORE:" + str(bleu_score) + "\n")
	#return mean, var

    def beam_translate(self, src_sent, beam_size=5):

        dynet.renew_cg()
        
        #wids = [[self.src_vocab[tok].i for tok in src_sent]]
        #embedded_seq = self.embed_batch_seq(wids)
        #embedded_seq_reverse = embedded_seq[::-1]
        #src_encodings, decoder_init = self.encode_batch_seq(embedded_seq, embedded_seq_reverse)
        
        wids = [self.src_vocab[tok].i for tok in src_sent]
        embedded_seq = self.embed_seq(wids)
        embedded_seq_rev = embedded_seq[::-1]
        src_encodings, decoder_init = self.encode_batch_seq(embedded_seq, embedded_seq_rev)

        W_s = dynet.parameter(self.W_s)
        b_s = dynet.parameter(self.b_s)
        W_h = dynet.parameter(self.W_h)
        b_h = dynet.parameter(self.b_h)
        W_y = dynet.parameter(self.decoder_w)
        b_y = dynet.parameter(self.decoder_b)

        completed_hypotheses = []
        decoder_init_cell = dynet.affine_transform([b_s, W_s, decoder_init])
        hypotheses = [Hypothesis(state=self.dec_lstm.initial_state([decoder_init_cell, dynet.tanh(decoder_init_cell)]),
                                 #state=self.dec_lstm.initial_state.add_input([decoder_init_cell, dynet.tanh(decoder_init_cell)]),
                                 y=[self.tgt_vocab['<s>'].i],
                                 ctx_tm1=dynet.vecInput(self.args.hidden_dim * 2),
                                 score=0., 
                                 alpha = [])]

        t = 0
        while len(completed_hypotheses) < beam_size and t < len(src_sent)*5:
            t += 1
            new_hyp_scores_list = []
            for hyp in hypotheses:
                y_tm1_embed = dynet.lookup(self.tgt_lookup, hyp.y[-1])
                x = dynet.concatenate([y_tm1_embed, hyp.ctx_tm1])

                hyp.state = hyp.state.add_input(x)
                h_t = hyp.state.output()
                ctx_t, alpha_t = self.attend(src_encodings, h_t, batch_size=1)

                assert abs(1 - np.sum(alpha_t.npvalue())) < 1e-2, 'sum(alpha_t) != 1'
                hyp.alpha.append(alpha_t.npvalue())

                # read_out = dynet.tanh(W_h * dynet.concatenate([h_t, ctx_t]) + b_h)
                read_out = dynet.tanh(dynet.affine_transform([b_h, W_h, dynet.concatenate([h_t, ctx_t])]))
                # y_t = W_y * read_out + b_y
                y_t = dynet.affine_transform([b_y, W_y, read_out])

                p_t = dynet.log_softmax(y_t).npvalue()
                #print(p_t.shape)
                hyp.ctx_tm1 = ctx_t

                # add the score of the current hypothesis to p_t
                new_hyp_scores = hyp.score + p_t
                new_hyp_scores_list.append(new_hyp_scores)

            live_nyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = np.concatenate(new_hyp_scores_list).flatten()
            new_hyp_pos = (-new_hyp_scores).argsort()[:live_nyp_num]
            prev_hyp_ids = new_hyp_pos / self.tgt_vocab.size
            word_ids = new_hyp_pos % self.tgt_vocab.size
            new_hyp_scores = new_hyp_scores[new_hyp_pos]
            new_hypotheses = []

            for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
                prev_hyp = hypotheses[prev_hyp_id]
                alpha = [np.copy(a) for a in prev_hyp.alpha]
                hyp = Hypothesis(state=prev_hyp.state,
                                 y=prev_hyp.y + [word_id],
                                 ctx_tm1=prev_hyp.ctx_tm1,
                                 score=hyp_score,
                                 alpha=alpha)

                if word_id == self.tgt_vocab.END_TOK.i:
                    completed_hypotheses.append(hyp)
                else:
                    new_hypotheses.append(hyp)

            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            assert len(hyp.y) == len(hyp.alpha) + 1, 'len(y) != len(alphas)'

        for hyp in completed_hypotheses:
            hyp.y = [self.tgt_vocab.i2t[idx] for idx in hyp.y]

        return sorted(completed_hypotheses, key=lambda x: x.score, reverse=True), src_encodings

