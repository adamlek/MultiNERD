import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from IPython import embed
from dataloader import format_labels, load_data, generate_batches
from transformers import AutoModel, AutoTokenizer
from model import NERClassifier
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from args import args
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from toolz import valmap

# Set seed for everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device(args.cuda_device)
writer = SummaryWriter('run')

def main():
    if args.system == 'A':
        exclude_labels = set([])
    else:
        exclude_labels = set(['BIO', 'CEL', 'EVE', 
                              'FOOD', 'INST', 'MEDIA', 
                              'PLANT', 'MYTH', 'TIME', 
                              'VEHI'])
        
    preprocess_func=lambda x: x
    
    dataset, idx2lbl_map, lf = load_data((('train', 'data/train_en.tsv'),
                                          ('dev', 'data/dev_en.tsv'),
                                          ('test', 'data/test_en.tsv')),
                                        exclude_labels=exclude_labels, 
                                        preprocess_func=preprocess_func)
    # Reverse label dictionary
    lbl2idx_map = dict(map(reversed, idx2lbl_map.items()))
    
    if args.save_or_load == 'save':
        base_model = AutoModel.from_pretrained(args.model_card, output_hidden_states = True)
        model = NERClassifier(base_model, len(idx2lbl_map))
    else:
        model = torch.load(f'models/system-{args.system}')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_card, add_prefix_space=True)
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-05)
    
    # Train and evaluate at each epoch
    logging_step = 0
    for e in range(args.epochs):
        if args.save_or_load == 'save':
            model.train()
            model, logging_step = train(model, tokenizer, opt, 
                                        dataset['train'], lbl2idx_map,
                                        logging_step, desc=f'Epoch {e}')
    
        model.eval()
        eval_model(model, tokenizer, dataset['dev'], lbl2idx_map, idx2lbl_map, desc='Dev')
        
    if args.do_test:
        eval_model(model, tokenizer, dataset['test'], 
                    lbl2idx_map, idx2lbl_map, 
                    eval_bi=True, produce_report=True)
        
    
def train(model, tokenizer, opt, dataset, lbl2idx_map, logging_step, desc='Train'):
    
    for batch in tqdm(generate_batches(dataset, batch_size=args.batch_size), 
                      total=int(len(dataset)/args.batch_size)+1,
                      desc=desc):
        tokenized_words = tokenizer(batch.words, 
                                    is_split_into_words=True, 
                                    padding=True, 
                                    return_tensors='pt')
        
        # Label propagation and decisions about how the B-[type]
        # label should be propagated across bpe-tokens. Currently,
        # the first bpe-token of the first word gets the B-[type] 
        # label. See `format_labels` for more details.
        labels = format_labels(batch.labels, tokenized_words, lbl2idx_map)
        output = model(tokenized_words.to(device))
        
        loss = F.cross_entropy(torch.transpose(output, 1, 2), 
                               labels, 
                               ignore_index=lbl2idx_map['<PAD>'])
        loss.backward()
        writer.add_scalar('loss', loss.item(), logging_step)
        
        # clip gradient norms so they don't get too large, 
        # value obtained mainly by intuition
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.5)
        
        opt.step()
        opt.zero_grad()
        
        logging_step += 1
    
    if args.save_or_load == 'save':
        torch.save(model, f'models/system-{args.system}')
    
    return model, logging_step
        
def eval_model(model, tokenizer, dataset, lbl2idx_map, idx2lbl_map, 
               eval_bi=False, produce_report=False, desc='Test'):
    pred_gold = []
    
    for batch in tqdm(generate_batches(dataset, batch_size=args.batch_size), 
                      total=int(len(dataset)/args.batch_size)+1,
                      desc=desc):
        tokenized_words = tokenizer(batch.words, 
                                    is_split_into_words=True, 
                                    padding=True, 
                                    return_tensors='pt')
        
        labels = format_labels(batch.labels, tokenized_words, lbl2idx_map)
                
        with torch.no_grad():
            output = model(tokenized_words.to(device))
            predictions = output.argmax(-1)
            
            # only consider the non-pad pairs
            pred_gold += list(filter(lambda x: x[1] != 0, 
                                     zip(predictions.flatten().tolist(), 
                                         labels.flatten().tolist())))
    
    if args.do_test and not eval_bi:
        # collapse labels: (B-PERS, I-PERS) -> PERS, ...
        pred_gold, idx2lbl_map, lbl2idx_map = collapse_labels(pred_gold, 
                                                              idx2lbl_map, 
                                                              lbl2idx_map)
    
    # Accuracy, Precision, Rcall and F1
    pred, gold = zip(*pred_gold)
    acc = np.mean([int(x==y) for x, y in pred_gold])
    f1 = f1_score(gold, pred, average='macro')
    precision = precision_score(gold, pred, average='macro')
    recall = recall_score(gold, pred, average='macro')
    # print as macro averaged metrics as latex row
    print(' & '.join(list(map(lambda x: str(np.round(x, 3)), 
                              [acc, precision, recall, f1])))+'\\\\')
    
    if produce_report:
        # create confusion matrix
        generate_confusion_matrix(pred, gold, idx2lbl_map)
        # generate per-label metrics
        per_label_scoring(pred, gold, lbl2idx_map)
        
    return 

def generate_confusion_matrix(gold, pred, idx2lbl_map):
    cm_matrix = confusion_matrix(gold, pred)
    # confusion matrix as percentages for easier visualization
    cm_matrix = cm_matrix/cm_matrix.astype(np.float64).sum(axis=1)
    
    # get labels for confusion matrix
    cm_labels = [idx2lbl_map[x] for x in range(len(idx2lbl_map)) 
                 if idx2lbl_map[x] not in ['<PAD>']]
    cm_df = pd.DataFrame(cm_matrix, cm_labels, cm_labels)
    sns.heatmap(data=cm_df, xticklabels=1, yticklabels=1, cmap='crest')
    plt.xticks(fontsize=7)
    plt.savefig(f'figs/confusionmatrix-system={args.system}.pdf')
    plt.clf()
    
def per_label_scoring(pred, gold, lbl2idx_map):
    # remove pad for classification report labels, a bit janky
    del lbl2idx_map['<PAD>']
    
    # get class-based precision, recall and f1-score
    clr = classification_report(gold, pred, output_dict=True, target_names=list(lbl2idx_map.keys()))
    rdf_df = list(chain.from_iterable([[[label, metric, v]
                                        for metric, v in metrics.items()]
                                    for label, metrics in clr.items()
                                    if label not in ['accuracy', 'macro avg', 'weighted avg', 'O']]))
    
    rdf_df = pd.DataFrame(rdf_df, columns=['label', 'metric', 'v'])
    # get rid of the 'support' for now, not relevant for our platting
    rdf_df.drop(rdf_df[rdf_df['metric']=='support'].index, inplace=True)
    g = sns.FacetGrid(rdf_df, col='metric')
    g.map(sns.barplot, 'label', 'v')
    for ax in g.axes[0]:
       ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.savefig(f'figs/label_metrics-system={args.system}.pdf')
    plt.clf()
        
def collapse_labels(pred_gold, idx2lbl_map, lbl2idx_map):
    
    new_labels = list(set(map(lambda x: x[2:] if x not in ['O', '<PAD>'] else x, 
                                  idx2lbl_map.values())))
        
    lbl2idx_map_update = {x:i for i, x in enumerate(new_labels)}
    idx2lbl_map_update = {i:x for i, x in enumerate(new_labels)}
    collapsed_pred_gold = []
    for p, g in pred_gold:
        p = idx2lbl_map[p]
        g = idx2lbl_map[g]
        
        if p not in ['O', '<PAD>']:
            p = p[2:]
        
        if g not in ['O', '<PAD>']:
            g = g[2:]
            
        p = lbl2idx_map_update[p]
        g = lbl2idx_map_update[g]
        collapsed_pred_gold.append((p, g))
    
    return collapsed_pred_gold, idx2lbl_map_update, lbl2idx_map_update

if __name__ == '__main__':
    main()
