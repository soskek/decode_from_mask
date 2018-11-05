from __future__ import print_function
import argparse
from itertools import zip_longest
import json

import chainer
from chainer import training
from chainer.training import extensions

import chain_utils
import nets
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--resume')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    vocab = json.load(open(args.vocab))

    """
    if args.snli:
        train = chain_utils.SNLIDataset(
            args.train_path, vocab)
        train = chain_utils.MaskingChainDataset(
            train, vocab['<mask>'], vocab, ratio=0.5)
        valid = chain_utils.SNLIDataset(
            args.valid_path, vocab)
        valid = chain_utils.MaskingChainDataset(
            valid, vocab['<mask>'], vocab, ratio=0.5)
    else:
        train = chain_utils.SequenceChainDataset(
            args.train_path, vocab, chain_length=1)
        train = chain_utils.MaskingChainDataset(
            train, vocab['<mask>'], vocab, ratio=0.5)
        valid = chain_utils.SequenceChainDataset(
            args.valid_path, vocab, chain_length=1)
        valid = chain_utils.MaskingChainDataset(
            valid, vocab['<mask>'], vocab, ratio=0.5)
    # Create the dataset iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)
    """
    # Prepare an biRNNLM model
    model = nets.DecoderModel(
        len(vocab), args.unit, args.layer, dropout=0.)

    if args.resume:
        print('load {}'.format(args.resume))
        chainer.serializers.load_npz(args.resume, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    inv_vocab = {i: w for w, i in vocab.items()}

    def translate(example):
        if len(example) == 4:
            source, target, zs, condition_source = example
            zs = [model.xp.array(zs)]
            conds = [model.xp.array(condition_source)]
        elif len(example) == 3:
            source, target, zs = example
            zs = [model.xp.array(zs)]
            conds = None
        elif len(example) == 2:
            source, target = example
            zs = None
            conds = None
        resultM = model.generate(
            [model.xp.array(source)],
            zs=zs,
            condition_xs=conds,
            sampling='argmax')
        resultR = model.generate(
            [model.xp.array(source)],
            zs=zs,
            condition_xs=conds,
            sampling='random',
            temperature=1.)

        target_sentence = [inv_vocab[y] for y in target[1:-1].tolist()]
        resultM_sentence = [inv_vocab[y] for y in resultM['out']]
        resultR_sentence = [inv_vocab[y] for y in resultR['out']]
        lens = [max(len(w1),
                    len(w2) if w2 is not None else 1,
                    len(w3) if w3 is not None else 1)
                for w1, w2, w3 in zip_longest(
            target_sentence, resultM_sentence, resultR_sentence)]
        source_sentence = [
            inv_vocab[x] if inv_vocab[x] != '<mask>' else '#' * lens[i]
            for i, x in enumerate(source[1:-1].tolist())]

        def format_by_length(sent):
            return ' '.join('{:^{length:}}'.format(w, length=l)
                            for w, l in zip(sent, lens))

        results = {}
        if condition_source is not None:
            results['label'] = chain_utils.inv_snli_label_vocab[int(zs[0][0])]
            results['condition'] = ' '.join(
                inv_vocab[wi] for wi in condition_source[1:-1].tolist())

        results['mask'] = format_by_length(source_sentence)
        results['greedy'] = format_by_length(resultM_sentence)
        results['sample'] = format_by_length(resultR_sentence)
        results['gold'] = format_by_length(target_sentence)
        return results

    # source, target, zs, condition_source = example
    example = [
        '<eos> <mask> <mask> is watching <mask> <mask> . <eos>',
        '<eos> two girls are watching blue flowers . <eos>',
        None,
        '<eos> two girls are watching blue flowers . <eos>',
    ]
    example = [np.array([vocab[w] for w in e.lower().split()], 'i')
               if e is not None else None
               for e in example]
    label = chain_utils.snli_label_vocab['entailment']
    example[2] = np.array([label] * (len(example[0]) - 1), 'i')

    samples = []
    for i in range(20):
        results = translate(example)
        samples.append(results['sample'])
    print(results['label'])
    print(results['condition'])
    print(results['mask'])
    print('|')
    print('v')
    print(results['greedy'])
    print('---')
    for s in sorted(samples):
        print(s)


if __name__ == '__main__':
    main()
