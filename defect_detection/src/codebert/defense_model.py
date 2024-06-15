# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class VictimModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, dropout_probability=0.):
        super(VictimModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_ids=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)

        return prob

    def get_reps(self, input_ids=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)

        return outputs