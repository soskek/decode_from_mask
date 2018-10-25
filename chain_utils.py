#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import io

import numpy as np
import progressbar

import chainer
from chainer import cuda
from chainer.dataset import dataset_mixin


def convert_sequence_chain(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return [to_device_batch([x[i] for x in batch])
            for i in range(len(batch[0]))]


def count_words_from_file(counts, file_path):
    bar = progressbar.ProgressBar()
    for l in bar(io.open(file_path, encoding='utf-8')):
        # TODO: parallel
        if l.strip():
            words = l.strip().split()
            for word in words:
                counts[word] += 1
    return counts


def make_chain_dataset(path, vocab, chain_length=2, max_toks=120,
                       get_last_only=False):
    dataset = []
    chain = []
    unk_id = vocab['<unk>']

    def make_array(chain):
        array_chain = []
        for i, words in enumerate(chain):
            if get_last_only and i != len(chain) - 1:
                continue
            tokens = []
            for word in words:
                tokens.append(vocab.get(word, unk_id))
            array_chain.append(np.array(tokens, 'i'))
        return array_chain

    bar = progressbar.ProgressBar()
    n_lines = sum(1 for _ in io.open(path, encoding='utf-8'))
    for line in bar(io.open(path, encoding='utf-8'),
                    max_value=n_lines):
        if not line.strip():
            if len(chain) >= chain_length:
                dataset.append(make_array(chain))
                chain = []
            continue
        words = line.strip().split()
        if len(words) > max_toks:
            words = words[:max_toks]
        words = ['<eos>'] + words + ['<eos>']
        chain.append(words)
    if len(chain) >= chain_length:
        dataset.append(make_array(chain))
    return dataset


class SequenceChainDataset(dataset_mixin.DatasetMixin):

    def __init__(self, path, vocab, chain_length=2,
                 get_last_only=False):
        self._dataset = make_chain_dataset(
            path,
            vocab=vocab,
            chain_length=chain_length,
            max_toks=50,
            get_last_only=get_last_only)
        self.chain_length = chain_length
        self._subchain_numbers = [
            len(chain) + 1 - chain_length
            for chain in self._dataset]
        self._length = sum(self._subchain_numbers)

        self._idx2subchain_keys = []
        chain_idx = 0
        for l in self._subchain_numbers:
            for i in range(l):
                self._idx2subchain_keys.append((chain_idx, i))
            chain_idx += 1

    def __len__(self):
        return self._length

    def get_subchain(self, subchain_keys):
        chain_idx, sub_idx = subchain_keys
        chain = self._dataset[chain_idx]
        subchain = chain[sub_idx: sub_idx + self.chain_length]
        return subchain

    def get_example(self, i):
        return self.get_subchain(self._idx2subchain_keys[i])

    def get_random(self):
        i = np.random.randint(len(self))
        return self.get_example(i)


class MaskingChainDataset(dataset_mixin.DatasetMixin):

    def __init__(self, dataset, mask, vocab, ratio=0.3, z_type='length'):
        self._dataset = dataset
        self.mask = mask
        self.ratio = ratio
        self.z_type = z_type
        self.inv_vocab = {i: w for w, i in vocab.items()}
        # TODO: varmask

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        chain = self._dataset.get_example(i)
        assert len(chain) == 1
        x = chain[0]
        # if chainer.config.train:
        rand = np.random.rand(x.size - 2) < self.ratio
        # keep bos and eos
        _x = x.copy()
        _x[1:-1] = np.where(rand, self.mask, _x[1:-1])
        if self.z_type == 'length':
            z = np.array(
                [len(self.inv_vocab[widx])
                 for widx in x.copy()[1:]], 'i')
            if chainer.config.train:
                z += np.random.randint(-2, 3, size=z.shape)
                # more/less than two words
            z = np.clip(z, 1, 40)
            if chainer.config.train:
                z = np.where(np.random.rand(*z.shape) > 0.5, z, 0)
                # half of tokens are unlabeled (0)
                # otherwise, length label is given, although little perturbed
            return (_x, x, z)
        else:
            return (_x, x)

    def get_random(self):
        i = np.random.randint(len(self))
        return self.get_example(i)
