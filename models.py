# encoding: utf-8
import torch


class ClsModel(torch.nn.Module):
    
    def __init__(self, emb_dim, vocab_size, input_dim, hidden_dim, bidirectional, num_layers, dropout, n_class):
        super(ClsModel, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_direc = 1
        if self.bidirectional:
            self.num_direc = 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_class = n_class
        self.emb = torch.nn.Embedding(self.vocab_size, self.emb_dim)
        self.gru = torch.nn.GRU(self.input_dim, self.hidden_dim, bidirectional=self.bidirectional,
                                num_layers=self.num_layers, dropout=self.dropout)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * self.num_direc, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.n_class)
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        emb = self.emb(inputs)
        batch_size = inputs.shape[0]
        h0 = torch.zeros([self.num_layers * self.num_direc, batch_size, self.hidden_dim])
        out, _ = self.gru(emb, h0)
        out = torch.transpose(out, 0, 1)
        out = out[:, -1, :]
        dense = self.dense(out)
        n_class = self.softmax(dense)
        return n_class


if __name__ == '__main__':
    inp = torch.randint(0, 10, [100, 10])
    outp = torch.randint(0, 2, [100, 1])
    model = ClsModel(5, 10, 5, 8, True, 2, 0.3, 2)
    cost = torch.nn.BCELoss()
    optimzer = torch.optim.Adam(model.parameters())

    for ep in range(5):
        running_loss = 0
        running_acc = 0
        datas = inp.view(10, 10, 10)
        ys = outp.view(10, 10, 1)
        for data, yi in zip(datas, ys):
            outs = model(data)
            batch_size = data.shape[0]
            one_hot = torch.zeros(batch_size, 2).scatter_(1, yi, 1)
            loss = cost(outs, one_hot)
            loss.backward()
            optimzer.step()
