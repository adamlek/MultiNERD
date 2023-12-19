import torch
import torch.nn as nn
import numpy as np
from args import args
from IPython import embed

device = torch.device(args.cuda_device)

class NERClassifier(nn.Module):
    def __init__(self, model, num_labels):
        super(NERClassifier, self).__init__()
        self.base_model = model
        self.emb_dim = self.base_model.embeddings.word_embeddings.weight.size(1)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Sequential(nn.Linear(self.emb_dim, 
                                                  self.emb_dim),
                                        self.dropout,
                                        nn.LeakyReLU(),
                                        nn.Linear(self.emb_dim, num_labels))
        
    def forward(self, input_data):
        output = self.base_model(**input_data)
        return self.classifier(output.last_hidden_state)