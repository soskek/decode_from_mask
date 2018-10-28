from __future__ import print_function
import argparse
from itertools import zip_longest
import json

import chainer
from chainer import training
from chainer.training import extensions

import chain_utils
import nets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gradclip', '-c', type=float, default=10,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--vocab', required=True)
    parser.add_argument('--train-path', '--train')
    parser.add_argument('--valid-path', '--valid')

    parser.add_argument('--resume')

    parser.add_argument('--labeled-dataset', '-ldata', default=None,
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--no-label', action='store_true')

    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--snli', action='store_true')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    vocab = json.load(open(args.vocab))

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

    print('#train =', len(train))
    print('#valid =', len(valid))
    print('#vocab =', len(vocab))

    # Create the dataset iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)

    # Prepare an biRNNLM model
    model = nets.DecoderModel(
        len(vocab), args.unit, args.layer, args.dropout)

    if args.resume:
        print('load {}'.format(args.resume))
        chainer.serializers.load_npz(args.resume, model)

    if args.labeled_dataset and not args.no_label:
        n_labels = len(set([int(v[1]) for v in valid]))
        print('# labels =', n_labels)
        model.add_label_condition_nets(n_labels, args.unit)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    iter_per_epoch = len(train) // args.batchsize
    print('{} iters per epoch'.format(iter_per_epoch))
    if iter_per_epoch >= 10000:
        log_trigger = (iter_per_epoch // 100, 'iteration')
        eval_trigger = (log_trigger[0] * 50, 'iteration')  # every half epoch
    else:
        log_trigger = (iter_per_epoch // 2, 'iteration')
        eval_trigger = (log_trigger[0] * 2, 'iteration')  # every epoch
    print('log and eval are scheduled at every {} and {}'.format(
        log_trigger, eval_trigger))

    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=chain_utils.convert_sequence_chain, device=args.gpu,
        loss_func=model.calculate_loss)

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(
        valid_iter, model,
        converter=chain_utils.convert_sequence_chain, device=args.gpu,
        eval_func=model.calculate_loss),
        trigger=eval_trigger)

    inv_vocab = {i: w for w, i in vocab.items()}

    @chainer.training.make_extension()
    def translate(trainer):
        example = valid.get_random()
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
            sampling='random')

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

        print('@______________________________')
        if condition_source is not None:
            print('@label ' +
                  chain_utils.inv_snli_label_vocab[int(zs[0][0])] + '\t')
            print('@cond: ' + ' '.join(
                inv_vocab[wi] for wi in condition_source[1:-1].tolist()))
        print('@MASK: ' + format_by_length(source_sentence))
        print('@PREm: ' + format_by_length(resultM_sentence))
        print('@PREr: ' + format_by_length(resultR_sentence))
        print('@GOLD: ' + format_by_length(target_sentence))
        print('@------------------------------')
    trainer.extend(
        translate, trigger=(100, 'iteration'))

    record_trigger = training.triggers.MinValueTrigger(
        'validation/main/perp',
        trigger=eval_trigger)
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)
    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)
    keys = ['epoch', 'iteration',
            'main/perp', 'validation/main/perp', 'elapsed_time']
    trainer.extend(extensions.PrintReport(keys),
                   trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=50))

    print('iter/epoch', iter_per_epoch)
    print('Training start')
    trainer.run()


if __name__ == '__main__':
    main()
