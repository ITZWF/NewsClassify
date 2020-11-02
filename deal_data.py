# encoding: utf-8
import os
import re
import sys
import jieba


class DealData(object):

    def __init__(self, **kwargs):
        self.dir = kwargs['dir']
        self.class_dict = {'forward': dict(), 'rev': dict()}
        self.fre_dict = dict()
        self.outdir = kwargs['outd']
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.fre_file = open(kwargs['fref'], mode='w', encoding='utf-8')
        self.class_file = open(kwargs['clsf'], mode='w', encoding='utf-8')

    def put_class_dict(self):
        for i, clss in enumerate(os.listdir(self.dir)):
            self.class_dict['forward'][clss] = i
            self.class_dict['rev'][i] = clss

    def deal_data(self):
        for cats in os.listdir(self.dir):
            cat_path = self.dir + '/' + cats
            out_cat_path = self.outdir + '/' + cats
            if not os.path.exists(out_cat_path):
                os.makedirs(out_cat_path)
            for item in os.listdir(cat_path):
                with open(cat_path + '/' + item, mode='r', encoding='utf-8') as f:
                    line = f.read()
                    jl = jieba.lcut(line)
                    for j in jl:
                        if j not in ['\n', '\t', ' ']:
                            if j not in self.fre_dict:
                                self.fre_dict[j] = 1
                            else:
                                self.fre_dict[j] += 1
                    nl = ' '.join(jl)
                    nl = re.sub(r'[\n\t ]+', ' ', nl)
                    with open(out_cat_path + '/' + item, mode='w', encoding='utf-8') as fw:
                        fw.write(nl)

    def __call__(self, *args, **kwargs):
        self.put_class_dict()
        self.class_file.write(str(self.class_dict))
        self.class_file.close()
        self.deal_data()
        sorted_fre_dict = sorted(self.fre_dict.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_fre_dict:
            self.fre_file.write(k + '\t' + str(v) + '\n')
        self.fre_file.close()


if __name__ == '__main__':
    [dir1, outd, fref, clsf] = sys.argv[1:]
    dd = DealData(dir=dir1, outd=outd, clsf=clsf, fref=fref)
    dd()
