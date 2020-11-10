import sys
import json

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import pandas as pd
import numpy as np
import torch

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

from sympy.core import symbols
# from sympy.abc import *
from sympy.logic.boolalg import to_cnf, Equivalent
from sympy.logic import simplify_logic
from sympy import solve, bool_map

# f = ~(((((v&v)>>i)|((i>>i)|v))>>v))
# f = to_cnf(f)



# #### Trained model below
# config = BertConfig('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/bert_prop_v2_99k/bert_config.json')
# model = BertForSequenceClassification(config, num_labels=3)
# model.load_state_dict(torch.load('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/bert_prop_v2_99k/pytorch_model.bin'))
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # kb = 'Christopher visited Saint Vincent & the Grenadines.'
# # hyp = 'Christopher did not visit Saint Vincent & the Grenadines. Christopher visited Saint Vincent & the Grenadines or Christopher did not visit Saint Vincent & the Grenadines.'
# # kb = 'Fred visited Qatar or Raymond visited Equatorial Guinea. Raymond visited Equatorial Guinea or Fred did not visit Qatar or Raymond did not visit Equatorial Guinea.'
# # hyp = 'Fred visited Qatar. Raymond visited Equatorial Guinea.'

# kb = "John visited France or Mark did not visit Spain. John did not visit France."
# hyp = "Mark visited in Spain."

# sent_1 = tokenizer.tokenize(kb)
# sent_2 = tokenizer.tokenize(hyp)
# tok = ['[CLS]'] + sent_1 + ['[SEP]'] + sent_2 + ['[SEP]']
# tok = tokenizer.convert_tokens_to_ids(tok)
# seg = [0]*(len(sent_1)+2) + [1]*(len(sent_2)+1) + [0]*(128 - len(tok))
# tok += [0]*(128 - len(tok))
# mask = [0 if t==0 else 1 for t in tok]

# x = torch.tensor(tok).unsqueeze(0)
# msk = torch.tensor(mask).unsqueeze(0)
# seg = torch.tensor(seg).unsqueeze(0)
# logits = model(x, seg, msk).squeeze()
# print(f'FALSE: {logits[-1]:.3f} \t | \t TRUE: {logits[0]:.3f}')

### Regex substitions
import re

# # s = "i & u & (e | u) & (~i | ~j)"
# # s = "(r | ~a) & (r | ~m) & (r | ~y) & (o | r | ~a) & (o | r | ~m) & (o | r | ~y) & (j | ~r | ~y)"
# # s = "a & (o | r` | ~a)"
# s = "(g | ~t) & (t | ~t) & (~g | ~t)"
# count = 10       # Can set randomly

# # 1. A | B <-> ~(~A & ~B) (De Morgan's law)
# expr = re.compile(r"(~?[a-z]) \| (~?[a-z]) \|")
# s_1 = expr.sub(r"~(~\1 & ~\2) |", s, count)     # Main string replacement
# s_1 = re.sub(r"~~", "", s_1)      # Remove double negations
# # print(s_1)
# s_1 = re.sub(r"(~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\))) \| (~?[a-z])", 
#         r"\2 | \1", s_1, count)     # Reverse ordering of terms
# # print(s_1)

# # 2. A | B <-> ~A -> B
# print(s_1)
# x = '>>'
# s_1 = re.sub(r"(~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\))) \| (~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\)))", 
#         fr"~\1 {x} \2", s_1, count)     # Main string replacement
# print(s_1)
# s_1 = re.sub(r"~~", "", s_1)      # Remove double negations
# # print(s_1)

# s_2 = re.sub(r"~?([a-z]) \| ~?\1", r"", s, count)
# s_2 = re.sub(r" & \(\)", r"", s_2, count)
# print(s_2)


# print(simplify_logic(s), simplify_logic(s_2))
# print(bool_map(s, s_2)[1])
# print(to_cnf(s_1), '\t', to_cnf(s_2))

## Other subs
# s_2 = re.sub(r"\(~?(\(~?[a-z] [\||&] ~?[a-z]\))\)", r"\1", s_2)      # Remove unnecessary parentheses (working)




# a = '((r&z)&(((j&t)>>s)&(g&(z&~(((i>>u)|m))))))'
# x = 'g & i & r & z & ~m & ~u & (s | ~j | ~t)'
# y = '(t >> ~(~s & j)) & g & r & ~m & ~u & z & i'

x = '(e | ~g) & (e | ~j) & (g | ~g) & (g | ~j) & (j | ~g) & (j | ~j) & (q | ~g) & (q | ~j) & (s | ~g) & (s | ~j)'
y = '(j >> j) & (j >> s) & (e | ~j) & (e | ~g) & (s | ~g) & (j | ~g) & (g | ~j) & (q | ~j) & (g | ~g) & (q | ~g)'
# x = '((j>>j)|p|q) & p'
# print(bool_map(x, y))
# print(to_cnf(x))
import time
s = time.time()
print(bool_map(simplify_logic(x, force=True), simplify_logic(y, force=True)))
print(time.time() - s)
# print(to_cnf(y))


# ((r&z)&(((j&t)>>s)&(g&(z&~(((i>>u)|m))))))       g & i & r & z & ~m & ~u & (s | ~j | ~t)         (t >> ~(~s & j)) & g & r & ~m & ~u & z & i
# (((g&(((j|(u|e))|(f|z))&~(~(c))))&s)|y)          (c | y) & (g | y) & (s | y) & (e | f | j | u | y | z)   (~y >> s) & (~y >> c) & (g | y) & (~(~e & ~f) | ~(~j & ~u) | y | z)