import argparse
import json
import glob
import os
import sys
import re
import time

from random import shuffle, choice, sample
from math import ceil
import pandas as pd
import numpy as np
import multiprocessing as mp

from sympy.core import symbols
from sympy.logic.boolalg import to_cnf
from sympy import bool_map
from sympy.logic import simplify_logic

def dump_args(args: argparse.Namespace):
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

def load_args(path: str):
    with open(path, 'r') as f:
        arg_dct = json.loads(f.read())
    return argparse.Namespace(**arg_dct)

def parse_names_and_locs():
    with open('/vol/bitbucket/aeg19/semantic_fragments/data/boolean/test/train.json', 'r') as f:
        samples = [json.loads(l) for l in f.readlines()]
    
    all_names, all_locations = set(), set()
    for n, sample in enumerate(samples):
        # Parse sentence 1
        if "have only visited" in sample['sentence1']:
            names, locations = sample['sentence1'].split("have only visited")
        elif "has only visited" in sample['sentence1']:
            names, locations = sample['sentence1'].split("has only visited")
        else:
            raise ValueError(f"{n} {sample}")

        names = names.replace(' and ', ', ').strip()
        locations = locations.replace(' and ', ', ').strip()
        all_names.update(names.split(', '))
        all_locations.update(locations.split(', '))

        # Parse sentence 2
        names, locations = sample['sentence2'].split("didn't visit")
        names = names.replace(' and ', ', ').strip()
        locations = locations.replace(' and ', ', ').strip()
        all_names.update(names.split(', '))
        all_locations.update(locations.split(', '))

    with open('/vol/bitbucket/aeg19/semantic_fragments/ag_scripts/args/names_locations.txt', 'w') as f:
        output = {'names': list(all_names), 'locations': list(all_locations)}
        json.dump(output, f, indent=2)

class PropositionalFormulaeProcessor:
    class Decors:
        @staticmethod
        def timer(func):
            def wrapper(*args):
                start_time = time.time()
                res = func(*args)
                print(f'EXECUTION TIME: {time.time() - start_time:.3f}')
                return res
            return wrapper

    def load_formulae(self, dir_path, ext, variant):
        ''' Function to load train and val files from a user specified directory.

            Each line in the file is a tuple (A, B, E) where:
                - A and B are propositions
                - E is an integer (1 or 0) indicating whether A entails B (written A ⊧ B)

            Variant determines the type of processing applied to the formula. See
            processing_func() for details.
        '''
        data_path = os.path.join(dir_path, ext + '.txt')

        df = pd.read_csv(data_path, names=['sentence1', 'sentence2', 'gold_label', 'H1', 'H2', 'H3']).iloc[:, :3]

        # Clean "implies" symbol so is recognized by 
        for i in range(2):
            df.iloc[:,i] = df.iloc[:,i].str.replace('>', '>>')

        if variant == 1:
            df['flavour'] = np.zeros(len(df))
            df['count'] = np.zeros(len(df))
        else:
            # Compile regex
            self.init_re(variant)
            # Draw parameters for number and type of disturbances
            df['flavour'] = np.random.choice(4, len(df))
            df['count'] = np.random.choice(3, len(df))

        # Convert formula to strings in CNF
        print('Processing')
        df['sentence1'] = self.apply_processing(df[['sentence1','flavour', 'count']], variant)
        df['sentence2'] = self.apply_processing(df[['sentence2','flavour', 'count']], variant)

        # print(df)
        return df.iloc[:,:-2] if variant==1 else df

    @Decors.timer
    def apply_processing(self, data: pd.DataFrame, variant: int):
        ''' Use mutliprocessing to speed up the processing of the
            propositional logical formulae.
        '''
        n_proc = mp.cpu_count()
        chunk_sz = ceil(len(data) / n_proc)
        # Split column into chunks
        chunks = [data.iloc[chunk_sz*i: chunk_sz*(i+1), :].values.tolist() for i in range(n_proc)]
        assert sum(map(len, chunks)) == len(data)
        # Process each chunk in parallel
        with mp.Pool(processes=n_proc) as pool:
            proc_results = pool.starmap(self.func, zip(chunks, [variant]*n_proc))
        # Combine result
        processed = pd.concat(proc_results).reset_index(drop=True)
        assert len(processed) == len(data)
        
        return processed

    def processing_func(self, data, variant):
        ''' Function to process a propositional logical formula.
            Variants:
                - 1: return formula in CNF
                - 2: return formula in CNF but with additional 
                added complexity
        '''
        formula, flavour, count = data
        if variant == 1:
            return str(to_cnf(formula))
        elif variant == 2:
            cnf = to_cnf(formula)
            s = str(cnf)

            # Shuffle order of terms
            terms = s.split(' & ')
            shuffle(terms)
            s = ' & '.join(terms)

            if flavour < 1:
                ops = []    # Do nothing
            elif flavour < 2:
                ops = ['or-and', 'dub-neg', 'rev', 'impl', 'dub-neg']
            elif flavour < 3:
                ops = ['or-and', 'dub-neg', 'rev',]
            else:
                ops = ['impl', 'dub-neg']

            for op in ops:
                s = re.sub(*self.res[op], s, count)

            # Confirm logical operators haven't altered the semantics
            f = lambda x: x[0]==x[1]
            if all(map(f, bool_map(cnf, s)[1].items())):
                pass    # Normal case- formulae are equivalent
            elif all(map(f, bool_map(cnf, simplify_logic(s, force=True))[1].items())):
                pass    # Formulae are equivalent after simplification (required if a clause is (A | ~A))
            elif all(map(f, bool_map(simplify_logic(cnf, force=True), simplify_logic(s, force=True))[1].items())):
                raise ValueError(f'Simplification should not change things here. Formulae: \n'
                                f'CNF:\t{cnf}\ns:\t{s}')
            else:
                raise ValueError(f'Formulae are not equivalent. Formulae: \n'
                                f'CNF:\t{cnf}\ns:\t{s}')

            # try:
            #     assert all(map(lambda x: x[0]==x[1], bool_map(cnf, s)[1].items())), \
            #         f"{cnf} \t != \t {s}"
            # except: 
            #     assert sorted(bool_map(cnf, s)[1].keys()) == sorted(bool_map(cnf, s)[1].values())
            #     print(formula, '\t', cnf, '\t', s)
            #     print(bool_map(cnf, s))
            #     pass

            return s

    def init_re(self, variant):
        ''' Function to compile regex so they can be re-used.
        '''
        if variant != 1:
            self.res = {
                # A | B <-> ~(~A & ~B) (De Morgan's law)
                'or-and': [
                    re.compile(r"(~?[a-z]) \| (~?[a-z]) \|"),
                    r"~(~\1 & ~\2) |",
                ],
                # ~~A <-> A (remove double negation)
                'dub-neg': [
                    re.compile(r"~~"),
                    "",
                ],
                # A | B <-> B | A (Reverse ordering of terms)
                'rev': [
                    re.compile(r"(~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\))) \| (~?[a-z])"),
                    r"\2 | \1",
                ],
                # A | B <-> ~A -> B
                'impl': [
                    re.compile(r"(~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\))) \| (~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\)))"),
                    r"~\1 >> \2",
                ],
                # A | ~A <-> ""
                'reduce': [
                    # re.compile(r"(~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\))) \| (~?(?:[a-z]|\(~?[a-z] & ~?[a-z]\)))"),
                    # r"~\1 >> \2",
                ]
            }

    def func(self, x, variant):
        ''' Helper for multiprocessing to define function
            at a higher level.
        '''
        return pd.Series(map(self.processing_func, x, [variant]*len(x)))


class TemplateUtils:
    DISTRACTOR_TEMPLATES = {
        'vanilla1': lambda p1, p2, l1, l2, v1, v2: f" {p1} {v1} {l1}.",
        'neg1': lambda p1, p2, l1, l2, v1, v2: f" {p1} {v2} {l1}.",
        'disj1': lambda p1, p2, l1, l2, v1, v2: f" {p1} or {p2} {v1} {l1}.",
        'disj2': lambda p1, p2, l1, l2, v1, v2: f" {p1} {v1} {l1} or {l2}.",
        'neg_disj1': lambda p1, p2, l1, l2, v1, v2: f" {p1} or {p1} {v2} {l1}.",
        'neg_disj2': lambda p1, p2, l1, l2, v1, v2: f" {p1} {v2} {l1} or {l2}.",
    }

    @staticmethod
    def shuffle_sentence(sentence: str):
        ''' Split an input sentence into sub-rules
            (seperated by '.'), shuffle their order
            and return the new sentence.
        '''
        sentences = [s.strip() for s in sentence.split('.') if s]
        shuffle(sentences)
        return '. '.join(sentences) + '.'

    def convert_to_template_old(self, formula):
        ''' Converts a propositional logic formula to a natural language 
            knowledge base using templates.

            args:
                - formula: str. Propositonal logic formula in conjunctive normal form
            returns:
                - kb: str. Natural language knowledge base
        '''
        # Remove logical operators
        logops = ['&', '|']
        sep = '§§§'
        formula = [i for i in formula if i not in logops]
        names = self.names[:]
        locs = self.locations[:]

        # Create dictionary with all atoms and a name + location for each
        atom_dct = {}
        for term in formula:
            if term != sep:
                atom = (set(term) & set(string.ascii_lowercase)).pop()
                atom_dct[atom] = {
                    'name': names.pop(choice(range(len(names)))),
                    'loc': locs.pop(choice(range(len(locs)))),
                }

        # formula = TemplateUtils.reorder_formula(formula)        

        # print(formula); sys.exit()
        # Create knowledge base by iterating over formula and converting
        # to natural language using templates
        kb = []
        loc = list(atom_dct.values())[0]['loc']
        while formula:
            item = formula.pop(0)
            if item == sep:
                kb.append(item)
            elif not '(' in item:
                if '~' in item:
                    item = item.strip('~')
                    # kb.append(f"{atom_dct[item]['name']} {self.neg_verb} {atom_dct[item]['loc']}.")
                    kb.append(f"{atom_dct[item]['name']} {self.neg_verb} {loc}.")
                else:
                    # kb.append(f"{atom_dct[item]['name']} {self.pos_verb} {atom_dct[item]['loc']}.")
                    kb.append(f"{atom_dct[item]['name']} {self.pos_verb} {loc}.")
            else:
                or_items = [item.strip('(')]
                while True:
                    item = formula.pop(0)
                    or_items.append(item.strip(')'))
                    if ')' in item:
                        break
                
                # Convert disjunction clause into natural language KB
                # loc = atom_dct[or_items[0].strip('~')]['loc']

                # Split list into atoms and negated atoms
                pos = [i for i in or_items if '~' not in i]
                neg = [i.strip('~') for i in or_items if '~' in i]
                pos_sentence, neg_sentence = '', ''
                if pos:
                    pos_sentence = (' or '.join([atom_dct[p]['name'] for p in pos]) + 
                                        ' ' + self.pos_verb + (' or ' if neg else ''))
                if neg:
                    neg_sentence = (' or '.join([atom_dct[n]['name'] for n in neg]) + 
                                        ' ' + self.neg_verb)
                kb.append(f'{pos_sentence}{neg_sentence} {loc}.')

        return ' '.join(kb)

    def convert_to_template(self, formula):
        ''' Converts a propositional logic formula to a natural language 
            knowledge base using templates.

            args:
                - formula: str. Propositonal logic formula in conjunctive normal form
            returns:
                - kb: str. Natural language knowledge base
        '''
        # Remove logical operators
        logops = ['&', '|']
        sep = '§§§'
        formula = [i for i in formula if i not in logops]
        names = self.names[:]
        locs = self.locations[:]

        # Create dictionary with all atoms and a name + location for each
        atom_dct = {}
        for term in formula:
            if term != sep:
                atom = (set(term) & set(string.ascii_lowercase)).pop()
                atom_dct[atom] = {
                    'name': names.pop(choice(range(len(names)))),
                    'loc': locs.pop(choice(range(len(locs)))),
                }

        # Natural language template
        template = lambda literal: (
            f"{atom_dct[literal.strip('~')]['name']} "    # Name
            f"{self.neg_verb if '~' in literal else self.pos_verb} "    # Verb
            f"{atom_dct[literal.strip('~')]['loc']}."      # Location
        )

        # Create knowledge base by iterating over formula and converting
        # to natural language using templates
        kb = []
        while formula:
            item = formula.pop(0)
            if item == sep:
                kb.append(item)
            elif not '(' in item:
                kb.append(template(item))
            else:
                or_items = [item.strip('(')]
                while True:
                    item = formula.pop(0)
                    or_items.append(item.strip(')'))
                    if ')' in item:
                        break
                
                # Convert disjunction clause into natural language sentence
                or_sentence = ' or '.join([template(item).strip('.') for item in or_items]) + '.'
                kb.append(or_sentence)

        return ' '.join(kb)

    @staticmethod
    def reorder_formula(formula):
        ''' Reorder the formula so the OR clauses come first
        '''
        f = []
        while formula:
            item  = formula.pop(0)
            if '(' not in item:
                f.append(item)
            else:
                or_items = [item]
                while True:
                    item = formula.pop(0)
                    or_items.append(item)
                    if ')' in item:
                        print(f, or_items)
                        f = or_items + f
                        break
        return f

    def add_distractors(self, sentence: str, max_num: int=5):
        ''' Randomly generate and add distractor facts to a sentence.
        '''
        for i in range(choice(range(max_num))):
            # Choose names and locations from remaining pool
            person_1 = choice(self.names)
            person_2 = choice([n for n in self.names if n!=person_1])
            loc_1 = choice(self.locations)
            loc_2 = choice([l for l in self.locations if l!=loc_1])

            # Randomly choose a distractor template
            template = choice(list(self.DISTRACTOR_TEMPLATES.values()))

            # Create distractor facts
            distractor = template(person_1, person_2, loc_1, loc_2, self.pos_verb, self.neg_verb)

            # Append to original sentence:
            sentence += distractor

        return sentence


if __name__ == '__main__':
    path = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/logical-entailment-dataset'
    variant = 2
    # PropositionalFormulaeProcessor().load_formulae(path, 'validate', variant).to_csv(
    #     os.path.join(path, f'v{variant}', 'dev.csv'))
    PropositionalFormulaeProcessor().load_formulae(path, 'train', variant).to_csv(
        os.path.join(path, f'v{variant}', 'train.csv'))
