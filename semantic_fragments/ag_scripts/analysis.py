import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

def main():
    df = pd.read_csv('/vol/bitbucket/aeg19/logical_plms/semantic_fragments/_experiments/propositional_v2/propositional_v2_correct_results.txt', header=None)
    df.columns = ['a', 'sentence1', 'sentence2', 'label', 'correct']

    df['correct'] = df['correct'].str.strip(']').str.strip().map({'True': 1, 'False': 0})
    df['len'] = (df['sentence1'] + '  '+ df['sentence2']).apply(len)
    for s in ['if', 'and', 'unless', 'or', 'we', 'not']:
        df[s] = (df['sentence1'] + '  '+ df['sentence2']).apply(lambda x: s in x)
        df[s] = df[s].astype(int)
        corr = df[[s,'correct']].corr()
        cf = confusion_matrix(df[s], df['correct'], labels=[1, 0])
        print(s, '\n', cf, '\n')#, df[s].value_counts())

    # cf = confusion_matrix((df['len']>100).astype(int), df['correct'])
    corr = df[['len','correct']].corr()
    print('len', '\n', corr, '\n')


if __name__ == '__main__':
    main()