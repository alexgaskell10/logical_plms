import sys
import json
import pandas as pd
import numpy as np
import torch

# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

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


# #### Trained model below
# config = BertConfig('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/propositional_v2/bert_config.json')
# model = BertForSequenceClassification(config, num_labels=3)
# model.load_state_dict(torch.load('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/propositional_v2/pytorch_model.bin'))
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # kb = 'If Todd did not visit Mauritania then Dan visited South Africa. If Fernando did not visit United Arab Emirates then Todd visited Mauritania unless Dan visited South Africa. Bobby visited Mauritania'
# # hyp = 'if Fernando did not visit United Arab Emirates then Fernando did not visit United Arab Emirates. If Benjamin did not visit Peru then Jack visited China unless Sergio visited Sri Lanka or Bobby did not visit Mauritania or Dan did not visit South Africa unless Sergio did not visit Sri Lanka. Bobby visited Mauritania or Fernando visited United Arab Emirates'
# # 'ENTAILMENT', False

# kb = "Jenny has not visited Panama. Danielle has visited Kuwait. Kimberly has visited Saint Lucia. if Kimberly has visited Saint Lucia then Diane has visited Tanzania"
# hyp = "Diane has visited Tanzania"

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
# print(f'Length: {len(tok)}\t|\tFALSE: {logits[-1]:.3f}\t|\tTRUE: {logits[0]:.3f}')


from sympy.core import symbols
from sympy.logic.boolalg import to_cnf, Equivalent
from sympy.logic import simplify_logic
from sympy import solve, bool_map

### Regex substitions
import re

# s = "((~o >>_ r) | ~y) & ((~o >>_ r) | ~m)"
# x = ('John', 'Spain')
# y = ('Mark', 'France')
# verb = lambda lit: self.pos_verb if "~" in lit else self.neg_verb
# t1 = lambda x,y: f"if {x[0].strip('~')} {verb()} {x[1].strip('~')} then {y[0].strip('~')} {verb()} {y[1].strip('~')} unless"
# ex1 = re.compile(r"\((~?[a-z]) [^\s]+ (~?[a-z])\) \|")
f = 'p_ & q_ & r_ & ~p_ & q_ & ~r_ & ~p_ & ~q_ & ~r_ & (p >>_ q) & s_ & t_'
# s = re.sub(r"(~?[a-z])_ & (~?[a-z]_)", lambda match: print(match.groups()), f)
# s = re.sub(r"(~?[a-z])_ & ", lambda m: print(m.groups()), f)
s = re.findall(r"(~[a-z])_ & "*3, f)
print(s)

# print(re.compile(r"(~?[a-z])_(?: & (~?[a-z])_)+").groups)
# print(re.search(r"(~?[a-z])_(?: & (~?[a-z])_)+", f).groups())
# print(re.findall(r"(~?[a-z])_(?: & (~?[a-z])_)+", f))
# print(re.findall(r"([a-z])_ & ([a-z])_", f))

# Connectives: or/unless; and/but
# ((o >> r) | ~m): if john doesn't visit spain then mark visits portugal or james doesn't visit france
# (~(~g & ~y) | ~z): (it isnt the case that / we won't find that) john and jane dont visit spain or frank doesnt visits portugal
# (~(~p & ~t) | (p >> v)): it isnt the case that john and jane dont visit spain unless if mark visits france then peter visits belgium
# ((o >> r) | (~p >> ~t)):

# ~: doesn't visit; won't visit
# |: or/unless
# &: and/but
# >>: if a then b/b if a