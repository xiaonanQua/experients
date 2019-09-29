# coding: utf-8
# @Author: Huzi Cheng
# @Date: 2017.08.25
# model from
# https://github.com/pytorch/examples/tree/master/word_language_model

from __future__ import print_function

import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    """
    Rnn model
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers):
        """
        model:model.RNNModel(args.model, input_size, args.nhid, args.nlayers)
        """
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            print("RNN Type: {0}".format(rnn_type))
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            print('RNN Type Error')

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.decoder = nn.Linear(nhid, ninp)  # output_size == input_size
        # self.get_logprobablities = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        output, hidden = self.rnn(input, hidden)
        # seq_len, batch, hidden_size * num_directions
        output = output.view(output.size()[0], -1)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden  # seq

    def init_hidden(self, bsz):
        """Init the hidden cells' states before any trial.
        """
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
