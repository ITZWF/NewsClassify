# encoding: utf-8
import re
import torch
import jieba


def predict(model_path, input_text, reverse_dict, max_len, cls_dict_rev):
    model = torch.load(model_path)
    inputs = jieba.lcut(input_text)
    inputs = ' '.join(inputs)
    inputs = re.sub(r'[\n\t ]+', ' ', inputs)
    inputs = inputs.split(' ')
    inputs_ls = [reverse_dict['<START>']]
    for x in inputs:
        if x in reverse_dict:
            inputs_ls.append(reverse_dict[x])
        else:
            inputs_ls.append(reverse_dict['<UNK>'])
    inputs_ls.append(reverse_dict['<END>'])
    if len(inputs_ls) >= max_len:
        inputs_ls = inputs_ls[:max_len]
    else:
        for i in range(len(inputs_ls), max_len):
            inputs_ls.append(reverse_dict['<PAD>'])

    with torch.no_grad():
        model.eval()
        inputs_ls = torch.Tensor(inputs_ls).view(1, -1)
        out = model(inputs_ls)
        pred = torch.argmax(out, -1)
        aim = cls_dict_rev[pred.item()]
        print(aim)
