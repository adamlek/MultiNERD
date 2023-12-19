from collections import namedtuple, Counter
from toolz import take
import random
from itertools import groupby
from IPython import embed
from args import args
import torch

device = torch.device(args.cuda_device)
random.seed(args.seed)

def chain_from_iterable_dict(iterables, key):
    # chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
    for it in iterables:
        for element in it[key]:
            yield element

def class_weights(labels):
    num_labels = Counter(labels)
    sum_labels = len(num_labels)
    return {k:sum_labels/v for k, v in num_labels.items()}

def format_labels(labels, tokenized_words, lbl2idx_map):
    """
    Because words are translated into bpe-tokens the mapping of 
    tokens to labels is different thus, we need to reassign labels 
    to bpe-tokens. Because a label may be associated with several 
    bpe-tokens, the status of the "B-[type]" label is put into 
    question. Is the "B" for the first bpe-token, or for the first 
    word? In this implementation, this decision is made with the 
    argument args.bpe_as_beginning.
    """
    
    # Get word ids of the bpe-tokens
    labels_to_bpe = [tokenized_words.word_ids(x) for x in 
                     range(tokenized_words.input_ids.size(0))]
    
    # Because we map labels to the bpe-tokens from the 
    # tokenizer we also get padding! (yaay)
    labels = label_propagation(labels, labels_to_bpe)
    
    # If we wish the that all bpe-tokens of the first 
    # word should have the B-[type] label
    if args.bpe_as_beginning:
        labels = bpe_to_btype(labels)
    
    # Map labels to indexes and create a torch tensor
    return torch.tensor(label_to_idx(labels, lbl2idx_map), device=device)

def label_to_idx(labels, mapping):
    """
    translate a batch of string labels to index labels 
    """
    return [[mapping[x] for x in s] for s in labels]

def label_padding(labels, max_len, padding_token='<PAD>'):
    return [x+[padding_token]*(max_len-len(x)) for x in labels]

def label_propagation(labels, word_to_bpe):
    """
    propagate labels from words to bpe-tokens
    "the Com-pa-ny Na-me jac-ket was nice"
     O   B   B  B  I  I  O   O   O   O 
    """
    return [['<PAD>' if x is None else labels[i][x] for x in s] 
            for i, s in enumerate(word_to_bpe)]

def bpe_to_btype(_labels):
    """
    set only the first bpe-token of the named entity to B-[type]
    "the Com-pa-ny Na-me jac-ket was nice"
     O   B   I  I  I  I  O   O   O   O

    A tad bit inefficient, but it works.
    """
    labels = []
    for i in range(len(_labels)):
        labels_i = []
        # group consecutive labels of the same type togheter
        f = [(i, list(j)) for i, j in groupby(_labels[i])]
        for n, lbls in f:
            if 'B' in n:
                if len(lbls) > 1: 
                    # replace sequences with multiple B-[type] labels
                    lbls[1:] = [x.replace('B-', 'I-') for x in lbls[1:]]
            labels_i += lbls
        labels.append(labels_i)
    return labels

def load_data(paths, exclude_labels=set([]), preprocess_func=lambda x: x):
    dataset = {'train':[],
               'dev':[],
               'test':[]}
    label_mapping = set()
    
    for split, path in paths:
        lang = path.split('.')[0].split('_')[1]
        with open(path) as f:
            sentence = {'words':[], 'labels':[], 'lang':lang}
            for line in f.readlines():
                
                # new sentence
                if line == '\n': 
                    dataset[split].append(sentence)
                    sentence = {'words':[], 'labels':[], 'lang': lang}
                    continue
                
                line = line.rstrip().split('\t')
                
                # is it a named entity or not?
                _, word, *label = line
                label = label[0] # get actual label
                if '-' in label:
                    if label[2:] in exclude_labels:
                        label = 'O'
                
                # collect labels from the training set
                label_mapping.add(label)
                sentence['words'].append(preprocess_func(word))
                sentence['labels'].append(label)
    
    # make a dict out of the labels and add the pad 
    # token as idx=0
    label_mapping = dict(enumerate(sorted(label_mapping), start=1))
    label_mapping[0] = '<PAD>'
    
    # comupte label frequency from train data
    label_freq = Counter(chain_from_iterable_dict(dataset['train'], 'labels'))
    
    return dataset, label_mapping, label_freq
    
def generate_batches(dataset, batch_size=8):
    """
    generate batches as namedtuples
    """
    Batch = namedtuple('Batch', ['words', 'labels', 'langs'])
    datalen = len(dataset)
    
    # randomize the dataset
    random.shuffle(dataset)
    
    dataset = iter(dataset)
    # iterate over the dataset N+1 times, 
    # where n is a batch (we do +1 to account 
    # for the last batch which may contain fewer examples)
    for _ in range(int(datalen/batch_size)+1):
        # take batch_size of items from dataset iterator
        batch = list(take(batch_size, dataset))
        if batch:
            # unpack batch into a namedtuple of words, 
            # labels and langs (langs is for when more 
            # languages are added to the training regime)
            yield Batch(*zip(*[x.values() for x in batch]))
                
if __name__ == '__main__':
    d, l, _ = load_data((('train', 'train_en.tsv'),
                        ('dev', 'dev_en.tsv'),
                        ('test', 'test_en.tsv')))