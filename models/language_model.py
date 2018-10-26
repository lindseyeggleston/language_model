import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """A recurrent network that generates new text"""

    def __init__(self, params):
        """
        Layer Description
        -----------------
        embedding: this layer maps each index in range(params.vocab_size) to
            a params.embedding_dim vector
        rnn: applies params.num_layers number of rnn layers of type
            params.rnn_type (either LSTM, GRU, RNN, RNN_RELU, or RNN_TANH) on
            the sequential input and returns an output for each token in the
            sentence
        fc: a fully connected layer that converts the rnn output for each
            token to a distribution over params.vocab_size tags

        Parameters
        ----------
        params (Params): contains vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, and bidirectional
        """
        super(Net, self).__init__()

        self.num_layers = params.num_layers
        self.hidden_dim = params.hidden_dim
        self.rnn_type = params.rnn_type.upper()

        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        if self.rnn_type in {'LSTM', 'GRU', 'RNN'}:
            self.rnn = getattr(nn, self.rnn_type)(params.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.nlayers,
                            batch_first=True,
                            dropout=params.dropout,
                            bidirectional=params.bidirectional))
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError('''Invalid option for `rnn_type` was supplied,
                             options are {'RNN', 'LSTM', or 'GRU'}''')
            self.rnn = nn.RNN(params.embedding_dim,
                              hidden_size=params.hidden_dim,
                              num_layers=params.nlayers,
                              nonlinearity=nonlinearity,
                              batch_first=True,
                              dropout=params.dropout,
                              bidirectional=params.bidirectional)

        self.fc = nn.Linear(params.hidden_dim, params.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        """
        Parameter
        ---------
        input: (Variable) contains a batch of sentences, of dimension
            batch_size x seq_len, where seq_len is the length of the
            longest sentence in the batch. For sentences shorter than
            seq_len, the remaining tokens are padding tokens. Each row is a
            sentence with each element corresponding to the index of the token
            in the vocab.

        Returns
        -------
        a Variable of dimension batch_size * seq_len x vocab_size with the log
        probabilities of tokens for each token of each sentence.
        """
        x = self.embedding(input)  # dim: batch_size x seq_len x embedding_dim
        x, _ = self.rnn(x)        # dim: batch_size x seq_len x rnn_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        x = s.contiguous()

        # reshape the Variable so that each row contains one token
        x = s.view(-1, x.shape[2])  # dim: batch_size*seq_len x rnn_hidden_dim
        x = self.fc(x)              # dim: batch_size*seq_len x vocab_size

        return F.log_softmax(x, dim=1)  # dim: batch_size*seq_len x vocab_size
