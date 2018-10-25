#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse
import json
import warnings

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)

EOS = 0
UNK = 1
MASK = 2


def embed_seq_batch(embed, seq_batch, dropout=0., context=None):
    x_len = [len(seq) for seq in seq_batch]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(seq_batch, axis=0))
    ex = F.dropout(ex, dropout)
    if context is not None:
        ids = [embed.xp.full((l, ), i).astype('i')
               for i, l in enumerate(x_len)]
        ids = embed.xp.concatenate(ids, axis=0)
        cx = F.embed_id(ids, context)
        ex = F.concat([ex, cx], axis=1)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class NormalOutputLayer(L.Linear):

    def __init__(self, *args, **kwargs):
        super(NormalOutputLayer, self).__init__(*args, **kwargs)

    def output_and_loss(self, h, t, reduce='mean'):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce=reduce)

    def output(self, h, t=None):
        return self(h)


def prepare_attention(variable_list):
    """Preprocess of attention.
    This function pads variables which have diffirent shapes
    such as ``(sequence_length, n_units)``.
    This function returns concatenation of the padded variables
    and an array which represents positions where pads do not exist.
    """
    xp = chainer.cuda.get_array_module(*variable_list)
    batch = len(variable_list)
    lengths = [v.shape[0] for v in variable_list]
    max_len = max(lengths)
    variable_concat = F.pad_sequence(
        variable_list, length=max_len, padding=0.0)
    pad_mask = xp.ones((batch, max_len), dtype='f')
    for i, l in enumerate(lengths):
        pad_mask[i, l:] = 0
    assert variable_concat.shape == (
        batch, max_len, variable_list[0].shape[-1])
    return variable_concat, pad_mask


def split_without_pads(V, lengths):
    """Postprocess of attention.
    This function un-pads a variable, that is,
    removes padded parts and split it into variables which have
    diffirent shapes such as ``(sequence_length, n_units)``.
    This function returns a list of variables.
    """
    batch, max_len, units = V.shape
    # get [len(seq1), len(pad1), len(seq2), len(pad2), ...]
    lengths = np.array(lengths)
    lengths_of_seq_and_pad = np.stack([lengths, max_len - lengths], axis=1)
    stitch = lengths_of_seq_and_pad.reshape(-1)
    # get indices to be split
    if stitch[-1] == 0:
        stitch = stitch[:-1]
    stitch_split_ids = np.cumsum(stitch)[:-1]
    # get [seq1, pad1, seq2, pad2, ...]
    # pick variables at even indices
    split_V = F.split_axis(V.reshape(batch * max_len, units),
                           stitch_split_ids, axis=0)[::2]
    return split_V


def func_dim3(func, x):
    assert x.ndim == 3
    batch, length, units = x.shape
    x_dim2 = x.reshape(batch * length, units)
    h_dim2 = func(x_dim2)
    h_dim3 = h_dim2.reshape(batch, length, -1)
    return h_dim3


class AttentionMechanism(chainer.Chain):
    def __init__(self, n_units, n_att_units=None):
        super(AttentionMechanism, self).__init__()
        self.n_units = n_units
        if n_att_units:
            self.att_units = n_att_units
        else:
            self.att_units = n_units
        with self.init_scope():
            self.W_query = L.Linear(None, self.att_units)
            self.W_key = L.Linear(None, self.att_units)

    def __call__(self, qs, ks):
        """Applies attention mechanism.
        Args:
            qs (~chainer.Variable): Concatenated query vectors
                Its shape is (batchsize, n_query, query_units).
            ks (~chainer.Variable): Concatenated key vectors.
                Its shape is (batchsize, n_key, key_units).
        Returns:
            ~chainer.Variable: Weighted sum of `ks`.
                The weight is computed by a learned function
                of keys and queries.
        """
        concat_Q, q_pad_mask = prepare_attention(qs)
        batch, q_len, q_units = concat_Q.shape
        Q = func_dim3(self.W_query, concat_Q)

        concat_K, k_pad_mask = prepare_attention(ks)
        batch, k_len, k_units = concat_K.shape
        K = func_dim3(self.W_key, concat_K)

        QK_dot = F.batch_matmul(Q, K, transb=True)
        assert QK_dot.shape == (batch, q_len, k_len)
        # ignore attention weights where padded values are used
        QK_pad_mask = q_pad_mask[:, :, None] * k_pad_mask[:, None, :]
        assert QK_pad_mask.shape == (batch, q_len, k_len)
        # padded parts are replaced with -1024,
        # making exp(-1024) = 0 when softmax
        minus_infs = self.xp.full(QK_dot.shape, -1024., dtype='f')
        QK_dot = F.where(QK_pad_mask.astype('bool'),
                         QK_dot,
                         minus_infs)
        QK_weight = F.softmax(QK_dot, axis=2)
        assert QK_weight.shape == (batch, q_len, k_len)

        # broadcast weight to be multiplied to vector
        QK_weight = F.broadcast_to(
            QK_weight[:, :, :, None],
            (batch, q_len, k_len, k_units))
        V = F.broadcast_to(
            concat_K[:, None, :, :],
            (batch, q_len, k_len, k_units))
        weighted_V = F.sum(QK_weight * V, axis=2)
        assert weighted_V.shape == (batch, q_len, k_units)
        q_lengths = [q.shape[0] for q in qs]
        split_weighted_V = split_without_pads(weighted_V, q_lengths)
        assert all(v.shape == (l, k_units) for l, v
                   in zip(q_lengths, split_weighted_V))
        return split_weighted_V


class DecoderModel(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5):
        super(DecoderModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            EncoderRNN = L.NStepBiLSTM
            DecoderRNN = L.NStepLSTM
            self.encoder = EncoderRNN(n_layers, n_units, n_units, dropout)
            self.decoder = DecoderRNN(n_layers, n_units, n_units, dropout)
            self.attention = AttentionMechanism(n_units, n_att_units=128)
            self.output = NormalOutputLayer(None, n_vocab)
            self.projection = L.Linear(None, n_units // 2)
            # self.mlp = chainer.Sequential(
            #    L.Highway(n_units),
            #    L.Highway(n_units))
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers
        self.use_bidirectional = True

    def encode_seq_batch(self, e_seq_batch, encoder):
        hs, cs, y_seq_batch = encoder(None, None, e_seq_batch)
        return hs, cs, y_seq_batch

    def __call__(self, xs, ys):
        # Prepare auto-regressive sequences
        # eos = self.xp.array([EOS], np.int32)
        # ys_in = [F.concat([eos, y], axis=0) for y in ys]
        # ys_out = [F.concat([y, eos], axis=0) for y in ys]
        ys_in = [y[:-1] for y in ys]
        ys_out = [y[1:] for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed, xs)
        eys = sequence_embed(self.embed, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, enc_os = self.encoder(None, None, exs)
        if self.use_bidirectional:
            # In NStepBiLSTM, cells of rightward LSTMs
            # are stored at odd indices
            hx = hx[::2]
            cx = cx[::2]
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)

        if hasattr(self, 'attention'):
            vs = self.attention(os, enc_os)
            concat_vs = F.concat(vs, axis=0)
            concat_os = F.concat(
                [concat_os, concat_vs], axis=1)
            concat_os = self.projection(concat_os)

        allloss = self.output.output_and_loss(
            concat_os, concat_ys_out, reduce='no')
        mask = self.xp.concatenate(
            [x[1:] for x in xs], axis=0) == MASK
        loss = F.sum(mask * allloss) / batch
        # loss = F.sum(self.output.output_and_loss(
        #    concat_os, concat_ys_out, reduce='no')) / batch

        chainer.report(
            {'loss': loss.data}, self)
        # n_words = concat_ys_out.shape[0]
        n_words = sum(mask)
        perp = self.xp.exp(
            loss.data * batch / n_words)
        chainer.report(
            {'perp': perp}, self)
        coef = (1. - mask) * 0.1
        loss = loss + F.sum(coef * allloss) / batch
        return loss

    def calculate_loss(self, batch_chain):
        xs = batch_chain[0]
        ys = batch_chain[1]
        return self(xs, ys)

    def generate_from_chain(self, chain,
                            max_length=100, sampling='random', temperature=1.):
        xs = chain[0]
        return self.generate(xs, max_length, sampling, temperature)

    def generate(self, xs, max_length=100, sampling='argmax', temperature=1.):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            exs = sequence_embed(self.embed, xs)
            h, c, enc_os = self.encoder(None, None, exs)
            if self.use_bidirectional:
                h = h[::2]
                c = c[::2]
            ys = self.xp.full(batch, EOS, np.int32)
            result = []
            for i in range(max_length):
                eys = self.embed(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)

                if hasattr(self, 'attention'):
                    vs = self.attention(ys, enc_os)
                    cvs = F.concat(vs, axis=0)
                    cys = F.concat([cys, cvs], axis=1)
                    cys = self.projection(cys)

                wy = self.output.output(cys)
                if sampling == 'random':
                    wy /= temperature
                    wy += self.xp.random.gumbel(size=wy.shape).astype('f')
                    ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                elif sampling == 'argmax':
                    ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                else:
                    raise ValueError
                result.append(ys)

        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS tags
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        result = {}
        result['out'] = outs[0]
        result['in'] = xs[0]
        return result

    """
    def predict(self, xs, labels=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            t_out_concat = self.encode(xs, labels=labels, add_original=0.)
            prob_concat = F.softmax(self.output.output(t_out_concat)).data
            x_len = [len(x) for x in xs]
            x_section = np.cumsum(x_len[:-1])
            ps = np.split(cuda.to_cpu(prob_concat), x_section, 0)
        return ps
    """


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs
