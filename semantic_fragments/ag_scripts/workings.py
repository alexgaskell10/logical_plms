import sys
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

if False:
    path = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/generate_challenge/brazil_fragments/counting/test/challenge_dev.tsv'
    df = pd.read_csv(path, header=None, sep='\t')
    df.columns = ['id', 'sentence1', 'sentence2', 'label']
    df = df.loc[df.label != 'NEUTRAL', :]
    print(df.label.value_counts())

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

# from sklearn.metrics import confusion_matrix

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# res = pd.read_csv('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/bert_prop_v2_99k/propositional_correct_results.txt', header=None)
# res.columns = ['a', 'b', 'c', 'd', 'result']
# res['result'] = res.result.str.strip(']')
# res['sentence'] = res.b + res.c
# res['tok'] = res.sentence.apply(tokenizer.tokenize)
# res['len'] = res.tok.apply(len)

# cf = confusion_matrix((res.len < 125).astype(str), res.result.astype(str))[-2:, :2]

# # print(res)
# print((res.len < 125).astype(str).value_counts(), res.result.astype(str).value_counts())
# print(cf / np.sum(cf, axis=1).reshape(2, 1))

# tokenizer.tokenize('~ a ')
# for ext in ['dev', 'train']:
#     res = pd.read_csv(f'/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/logical-entailment-dataset/{ext}.csv').iloc[:, 1:]

#     # res['sentence1'] = res['sentence1'].str.replace("~", "not ").str.replace("|", "or").str.replace("&", "and")
#     # res['sentence2'] = res['sentence2'].str.replace("~", "not ").str.replace("|", "or").str.replace("&", "and")
#     res['gold_label'] = res['gold_label'].map({0: 'CONTRADICTION', 1: 'ENTAILMENT'})
#     res.to_csv(f'/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/proplog/v1/challenge_{ext}.tsv', sep='\t', header=None)


#### Trained model below
d = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/proplog/v3/'
# d = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/propositional_v3/'
config = BertConfig(os.path.join(d, 'bert_config.json'))
model = BertForSequenceClassification(config, num_labels=3)
model.load_state_dict(torch.load(os.path.join(d, 'pytorch_model.bin')))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# kb = "Edgar, Mark and Jane visited Luxembourg. Edgar, Jane and Mark didn't visit Luxembourg if John visited France. John visited France."
# hyp = "Jane visited Luxembourg."
equivs = [
    # 1
    ['(p&q)','(q&p)', True],
    ['(p&q)','~(p&q)', False],
    ['(p&q)','(~(p)&q)', False],
    # 21
    ['(p>q)','(~(p)|q)', True],
    ['(p>q)','(~(p)|~(q))', False],
    ['(p>q)','(p|q)', False],
    ['(p>q)','~(~(p)|q)', False],
    # 22
    ['(p>q)','~(p&~(q))', True],
    ['(p>q)','~(~(p)&~(q))', False],
    ['(p>q)','(p&~(q))', False],
    ['(p>q)','~(p&q)', False],
    # 23
    ['~(p>q)','(p&~(q))', True],
    ['~(p>q)','(~(p)&~(q))', False],
    ['~(p>q)','~(p&~(q))', False],
    ['~(p>q)','(p&q)', False],
]
# equivs = [
#     # ["John visited France and Jane visited Spain.", "John didn't visited France and Jane visited Spain.", False],
#     ["If John visited France then Jane visited Spain.", "John didn't visit France or Jane visited Spain.", False],
#     ["If John visited France then Jane visited Spain.", "John didn't visit France or Jane didn't visit Spain.", False],
#     ["If John visited France then Jane visited Spain.", "John visited France or Jane visited Spain.", False],
#     ["If John visited France then Jane visited Spain.", "It isn't the case that John didn't visit France or Jane visited Spain.", False],
# ]
for e in equivs:
    kb, hyp, label = e
    sent_1 = tokenizer.tokenize(kb)
    sent_2 = tokenizer.tokenize(hyp)
    tok = tok_ = ['[CLS]'] + sent_1 + ['[SEP]'] + sent_2 + ['[SEP]']
    tok = tokenizer.convert_tokens_to_ids(tok)
    seg = [0]*(len(sent_1)+2) + [1]*(len(sent_2)+1) + [0]*(128 - len(tok))
    tok += [0]*(128 - len(tok))
    mask = [0 if t==0 else 1 for t in tok]

    x = torch.tensor(tok).unsqueeze(0)
    msk = torch.tensor(mask).unsqueeze(0)
    seg = torch.tensor(seg).unsqueeze(0)
    logits = model(x, seg, msk).squeeze()
    probs = F.softmax(logits, dim=0)
    print(f'Length: {len(tok_)}\t|\tFALSE: {logits[-1]:.3f}\t|\tTRUE: {logits[0]:.3f}',
        f'\t|\tP(TRUE): {probs[0]:.3f}',
        f'\t|\tLabel: {label}')


# from sympy.core import symbols
# from sympy.logic.boolalg import to_cnf, Equivalent
# from sympy.logic import simplify_logic
# from sympy import solve, bool_map

# ### Regex substitions
# import re

# f = 'p_ & q_ & r_ & ~p_ & q_ & ~r_ & ~p_ & ~q_ & ~r_ & (p >>_ q) & s_ & t_'
# s = re.findall(r"(~[a-z])_ & "*3, f)
# print(s)


# in_dir = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/logical-entailment-dataset'
# out_dir = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/proplog/v3'

# names = [('train.txt', 'challenge_train.tsv'), ('validate.txt', 'challenge_dev.tsv')]
# for i in [0, 1]:
#     df = pd.read_csv(os.path.join(in_dir, names[i][0]), header=None)
#     df = df.iloc[:, :3]
#     df.iloc[:, -1] = df.iloc[:, -1].map({0: 'CONTRADICTION', 1: 'ENTAILMENT'})
#     df.to_csv(os.path.join(out_dir, names[i][1]), sep='\t', header=None)