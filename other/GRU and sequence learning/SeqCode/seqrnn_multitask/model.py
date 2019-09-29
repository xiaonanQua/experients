# coding: utf-8
# @Author: Huzi Cheng
# @Date: 2017.08.25
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable


class GRUNetwork(nn.Module):
    """
    GRU network model
    """

    def __init__(self, ninp, nhid, bsz, nlayers, lr, cuda_enabled=False, init_noise_amp=0.0, bg_noise_amp=0.0):
        super(GRUNetwork, self).__init__()
        self.rnn = nn.GRU(ninp, nhid, num_layers = nlayers)
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.batch_size = bsz
        self.decoder = nn.Linear(nhid, ninp)  # output_size == input_size
        # self.get_logprobablities = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.init_noise_amp = init_noise_amp
        self.bg_noise_amp = bg_noise_amp
        self.cuda_enabled = cuda_enabled
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduce=True)
        if self.cuda_enabled:
            torch.cuda.manual_seed_all(int(time.time()))
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            torch.manual_seed(int(time.time()))
            self.init_hidden = self.init_hidden_cpu

    def forward(self, input, hidden, bsz = None):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        bsz = self.batch_size if bsz == None else bsz
#        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, hidden)
        hidden = self.relu(hidden)
        output = self.relu(output)
        output = output.view(bsz, -1) 
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden # seq
    
        
    def init_hidden_cpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_() + self.get_noise(self.nlayers, bsz, self.nhid))

    def init_hidden_gpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda())

    def get_noise(self, layers, bsz, nhid, ):
        return torch.randn(layers, bsz, self.nhid) * self.init_noise_amp
    
    
class LSTMNetwork(nn.Module):
    """
    LSTM network model
    """

    def __init__(self, ninp, nhid, bsz, nlayers, lr, cuda_enabled=False, init_noise_amp=0.0, bg_noise_amp=0.0):
        super(LSTMNetwork, self).__init__()
        self.rnn = nn.LSTM(ninp, nhid, num_layers = nlayers)
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.batch_size = bsz
        self.decoder = nn.Linear(nhid, ninp)  # output_size == input_size
        # self.get_logprobablities = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_noise_amp = init_noise_amp
        self.bg_noise_amp = bg_noise_amp
        self.cuda_enabled = cuda_enabled
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.MSELoss(reduce=True)

        if self.cuda_enabled:
            torch.cuda.manual_seed_all(int(time.time()))
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            torch.manual_seed(int(time.time()))
            self.init_hidden = self.init_hidden_cpu

    def forward(self, input, hidden, bsz = None):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        bsz = self.batch_size if bsz == None else bsz
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, hidden)
        hidden_ = (self.relu(hidden[0]), hidden[1])
        output = self.relu(output)
        output = output.view(bsz, -1) 
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden_
    
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def init_hidden_cpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def init_hidden_gpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()), 
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()))

    def get_noise(self, layers, bsz, nhid, ):
        return torch.randn(layers, bsz, self.nhid) * self.init_noise_amp

class RNNNetwork(nn.Module):
    """
    RNN network model
    """

    def __init__(self, ninp, nhid, nlayers, cuda_enabled=False, init_noise_amp=0.0, bg_noise_amp=0.0):
        super(RNNNetwork, self).__init__()
        self.rnn = nn.RNN(ninp, nhid, nlayers, cuda_enabled, bg_noise_amp=bg_noise_amp)
        self.nhid = nhid
        self.nlayers = nlayers
        self.decoder = nn.Linear(nhid, ninp)  # output_size == input_size
        # self.get_logprobablities = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.init_noise_amp = init_noise_amp
        self.bg_noise_amp = bg_noise_amp
        self.cuda_enabled = cuda_enabled

        if self.cuda_enabled:
            torch.cuda.manual_seed_all(int(time.time()))
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            torch.manual_seed(int(time.time()))
            self.init_hidden = self.init_hidden_cpu
            
    def forward(self, input, hidden):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        output, hidden = self.rnn(input, hidden)
        output = output.view(output.size()[0], -1)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden# seq

    def init_hidden_cpu(self, bsz):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_() + self.get_noise(self.nlayers, bsz, self.nhid))

    def init_hidden_gpu(self, bsz):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda())

    def get_noise(self, layers, bsz, nhid, ):
        return torch.randn(layers, bsz, self.nhid) * self.init_noise_amp



