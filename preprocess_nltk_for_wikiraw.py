import argparse
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', '-d', required=True)
args = parser.parse_args()

text = []
for i_line, l in enumerate(io.open(args.data, encoding='utf8')):
    if i_line % 100000 == 0:
        sys.stderr.write('{} lines end\n'.format(i_line))
    l = l.strip()
    if not l:
        print('')
        continue

    sents = sent_tokenize(l)
    for sent in sents:
        # tokens = word_tokenize(sent)
        print(sent.strip().lower())
