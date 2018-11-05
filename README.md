# Sentence from Masked Sentence

Scripts for training/running models to generate a natural language sentence from a masked sentence (c.f. [MaskGAN](https://arxiv.org/pdf/1801.07736.pdf), [Text infilling](https://openreview.net/pdf?id=r1zmVhCqKm), [etc.](https://arxiv.org/pdf/1805.06201.pdf)).

For example, given `he [###] another [###] as governor [###] [###] .`,
a trained model generated samples, `he was another lieutenant as governor of <unk>.`
or `he defined another seminary as governor of somerset.`
(btw, its original sentence is `he sought another term as governor in 1927.`).


## Usage

### Prepare dataset

You can use any dataset files, which have a tokenized sentence per line.

If you just want to test a model, one sample dataset is the Wikitext103 dataset.
You can use it as follows.

```
# download wikitext
sh prepare_rawwikitext.sh

# segment text by sentence boundaries
PYTHONIOENCODING=utf-8 python preprocess_nltk_for_wikiraw.py -d datasets/wikitext-103-raw/wiki.train.raw > datasets/wikitext-103-raw/wiki.train.tokens
PYTHONIOENCODING=utf-8 python preprocess_nltk_for_wikiraw.py -d datasets/wikitext-103-raw/wiki.valid.raw > datasets/wikitext-103-raw/wiki.valid.tokens
```

### Prepare vocabulary

```
python construct_vocab.py --data datasets/wikitext-103-raw/wiki.train.tokens -t 100 -s datasets/wikitext-103-raw/vocab.t100.json
```

### Train a model


```
mkdir logs
mkdir outs
python -u train.py -g 0 --train datasets/wikitext-103-raw/wiki.train.tokens --valid datasets/wikitext-103-raw/wiki.valid.tokens --vocab datasets/wikitext-103-raw/vocab.t100.json -u 256 --layer 1 --dropout 0.1 --batchsize 128 --lr 1e-3 --out outs/v1.u256 | tee logs/v1.u256
```

## Usage for Dataset with labels and condition sentences

SNLI is the dataset.

An generated example:
```
@label entailment
@cond: a woman , wearing a white shirt and green shorts , sitting on a rock in a beautiful body of water .
@MASK: the woman ###### ####### ######## ##### .
@PREm: the woman   is   wearing    a     shirt . (greedy generation)
@PREr: the woman enjoys  rocks    near   water . (sampling)
@GOLD: the woman   is     not   standing  up   .
```

Usage:

```
# Download
cd datasets
curl https://nlp.stanford.edu/projects/snli/snli_1.0.zip -o snli_1.0.zip
unzip snli_1.0.zip
cd ..

# Make a flattened sentence file
# e.g.
less datasets/snli_1.0/snli_1.0_train.txt | cut -f2 | cat <(less datasets/snli_1.0/snli_1.0_train.txt | cut -f3) | sed -E 's/\)//g' | sed -E 's/\(//g' > datasets/snli_1.0/snli_1.0_train.txt.sents
# construct vocab
python construct_vocab.py --data datasets/snli_1.0/snli_1.0_train.txt.sents -t 2 -s datasets/snli_1.0/vocab.t2.json

# Train
python -u train.py -g 0 --train datasets/snli_1.0/snli_1.0_train.txt --valid datasets/snli_1.0/snli_1.0_dev.txt --vocab datasets/snli_1.0/vocab.t2.json -u 512 --layer 1 --epoch 40 --dropout 0.2 --batchsize 128 --lr 1e-3 --out outs/snli.u512 --snli | tee logs/snli.u512
```


## License

MIT License. Please see the LICENSE file for details.
