# encoding: utf-8
import torch
from torch.utils.data import DataLoader

from create_dataset import ClsDataset
from models import ClsModel


def create_dict(cls_file, fre_file, vocab_size):
    fre_dict = {'<PAD>': 0, '<UNK>': 1, '<END>': 2}
    with open(cls_file, mode='r', encoding='utf-8') as f:
        cls_dict = eval(f.read())
    lines = open(fre_file, mode='r', encoding='utf-8').readlines()[:vocab_size]
    for m, line in enumerate(lines):
        item, _ = line.strip().split('\t')
        fre_dict[item] = m + 3
    fre_dict_rev = dict()
    for k, v in fre_dict.items():
        fre_dict_rev[v] = k
    return cls_dict, fre_dict, fre_dict_rev


if __name__ == '__main__':
    DIR = './'
    LR = 1e-3
    VOCAB_SIZE = 10000
    EPOCH = 10
    BATCH_SIZE = 1024

    cls_dicts, fre_dicts, fre_dicts_rev = create_dict('', '', VOCAB_SIZE)
    dataset = ClsDataset(DIR, cls_dicts, fre_dicts, 100)
    train_data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ClsModel(10, VOCAB_SIZE+3, 10, 16, True, 2, 0.3, len(cls_dicts.keys()))
    cost = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCH):
        run_loss = 0
        train_correct = 0
        for i, data in enumerate(train_data):
            x = data['text']
            y = data['label']
            one_hot = torch.zeros(BATCH_SIZE, 10).scatter_(1, y, 1)
            outs = model(x)
            pred = torch.argmax(outs, -1)
            loss = cost(outs, one_hot)
            optim.zero_grad()
            loss.backward()
            optim.step()
            run_loss += loss.item()
            train_correct += (pred == y.squeeze(1)).sum()
            if i % 10 == 0:
                print('epoch: %d, step: %d, loss: %.4f, acc: %.4f' % (epoch, i, loss, train_correct/10*BATCH_SIZE))
                run_loss = 0
                train_correct = 0
        torch.save(model, './models/cls-model-%s.pkl' % epoch)

