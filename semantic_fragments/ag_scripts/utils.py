import argparse
import json
import glob
import os
import sys
import re
import time
import string

from random import shuffle, choice, sample, uniform
from math import ceil
from collections import OrderedDict
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
            self.init_regex()
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

            # Apply equivalences to perturb formula
            if flavour < 1:
                ops = []    # Do nothing
            elif flavour < 2:
                ops = ['or-and', 'dub-neg', 'rev', 'impl', 'spaces', 'dub-neg']
            elif flavour < 3:
                ops = ['or-and', 'dub-neg', 'rev',]
            else:
                ops = ['impl', 'spaces', 'dub-neg',]

            for op in ops:
                s = re.sub(*self.res[op], s, count)

            # Ensure new and old formulae are semantically equivalent
            test_func = lambda bm: bm and all(map(lambda x: x[0]==x[1], bm[1].items()))
            new_formula = s.replace('_', '')
            try:
                bm = bool_map(cnf, new_formula)
            except:
                bm = None

            if test_func(bm): 
                pass    # Normal case- formulae are equivalent
            else:
                bm = bool_map(simplify_logic(cnf, force=True), simplify_logic(new_formula, force=True))
                if test_func(bm): 
                    pass    # Formulae are equivalent after simplification (required if a clause is (A | ~A))
                else:
                    # raise ValueError(f'Formulae are not equivalent. Formulae: \n'
                    #                 f'CNF:\t{cnf}\ns:\t{new_formula}')
                    print(f'Formulae are not equivalent. Formulae: \nCNF:\t{cnf}\ns:\t{new_formula}')
            
            return s

    def init_regex(self):
        ''' Function to compile regex so they can be re-used.
        '''
        self.res = {
            # A | B <-> ~(~A & ~B) (De Morgan's law)
            'or-and': [
                re.compile(r"(~?[a-z]) \| (~?[a-z]) \|"),
                r"~(~\1 &_ ~\2) |",
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
                r"(~\1 >>_ \2)",
            ],
            # ((A | B)) <-> (A | B) (Remove excess spaces)
            'spaces': [
                re.compile(r"\((\(~?[a-z] (?:[^\s]+ ~?[a-z])+\))\)"),
                r"\1",
            ],
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

    def convert_to_template_v1(self, formula):
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

    def convert_to_template_v2(self, formula, sep=' §§§ ', generic=False):
        # Remove logical operators
        names = self.names[:]
        locs = self.locations[:]
        # formula = 's & a & p & (~(~d &_ l) | ~s) §§§ (~(~l &_ p) | ~y) & (~(~a &_ ~l) | ~p) & ((p >>_ l) | ~q)'.split()

        # Preprocessing
        if True:
            formula = _formula = ' '.join(formula)
            formula = formula.replace(' & ', ' + ')
            # Reorder so that negated / positive literals are consecutive
            formula = [f.split(' + ') for f in formula.split(sep)]
            formula = __formula = ' + '.join(sorted(formula[0], reverse=True)) + f" {sep} " + ' + '.join(sorted(formula[1], reverse=True))

        # Assign a name and location to each atom
        index = 0
        atom_dct = {}
        for term in formula:
            atom = (set(term) & set(string.ascii_lowercase))
            if atom and list(atom)[0] not in atom_dct:
                if generic:
                    item = {'name': f'person-{index}', 'loc': f'location-{index}'}
                    atom_dct[atom.pop()] = item
                    index += 1
                else:                    
                    # Ensure that two atoms don't have same assignment
                    while True:
                        item = {'name': choice(names), 'loc': choice(locs)}
                        if item not in atom_dct.values():
                            break
                    atom_dct[atom.pop()] = item

        ## Natural language template
        # Helper units to be used by subsequent functions
        if True:
            and_aux1 = lambda: choice(["it isn't the case that", "we won't find that"])
            and_aux2 = lambda: choice([' and ', ' but ', ', ', '. '])
            pos_verb = lambda: choice(["has visited", "visited", "went to", "holidayed in", "spent time in"])
            neg_verb = lambda: choice(["hasn't visited", "did not visit", "didn't go to", "hasn't holidayed in", "didn't spent time in"])
            # Base units
            atom = lambda p: atom_dct[p.strip('~')]
            name = lambda p: atom(p)['name']
            verb = lambda p: neg_verb() if '~' in p else pos_verb()
            loc = lambda p: atom(p)['loc']
            base = lambda p: f"{name(p)} {verb(p)} {loc(p)}"
            # Sentences (built by combining above units)
            if_base = lambda p,q: choice([f"if {base(p)} then {base(q)}", f"{base(q)} if {base(p)}"])
            if_unless = lambda p,q: f"{if_base(p,q)} unless"
            or_base = lambda _: np.random.choice(["or", "unless"], 1, p=(0.9, 0.1)).item()
            and_neg = lambda p,q: f"{and_aux1()} {base(p)} and {base(q)}"
            and_unless = lambda p,q: f"{and_aux1()} {base(p)} and {base(q)} unless"
            and_base = lambda p: f"{base(p)}{and_aux2()}"
            def and_chain(*args):
                # Overwrite the locations of later atoms with the first atom's location
                for m in args[1:]:
                    atom_dct[m.strip('~')] = {'name': name(m), 'loc': loc(args[0])}
                return f"{name(args[0])}, {name(args[1])} and {base(args[2])}. "

        # Regexs: (m, r, p)
        #   m: matching expression
        #   r: replacement expression
        #   p: probability of applying substitution
        # ORDER MATTERS HERE
        exprs = [
            (re.compile(r"([^~][a-z])_ \+ "*3), and_chain, 0.5),   # {p, q, r, ...}
            (re.compile(r"(~[a-z])_ \+ "*3), and_chain, 0.5),   # {~p, ~q, ~r, ...}
            (re.compile(r"\((~?[a-z])_ >>_ (~?[a-z])_\) \|"), if_unless, 1.0),   # {(p -> q | r ...), ....}
            (re.compile(r"\((~?[a-z])_ >>_ (~?[a-z])_\)"), if_base, 1.0),    # {p -> q, ....}
            (re.compile(r"~\((~?[a-z])_ &_ (~?[a-z])_\) \|"), and_unless, 1.0),  # {(~(p & q) | r ...), ...}
            (re.compile(r"~\((~?[a-z])_ &_ (~?[a-z])_\)"), and_neg, 1.0),    # {~(p & q), ...}
            (re.compile(r"(~?[a-z])_ \+ "), and_base, 1.0),   # {p, ...}
            (re.compile(r"(~?[a-z])_"), base, 1.0),  # {p, ...}
            (re.compile(r"(\|)"), or_base, 1.0),     # {(p | q ...), ...}
        ]

        # Preprocessing
        if True:
            formula = re.sub(r"([a-z])", r"\1_", formula)
            formula = re.sub(r"~~", r"", formula)
            formula = re.sub(r"\((\([^\)]+\))\)", r"\1", formula)
            
        # Perform conversion:
        # use regex to identify subformulae and convert these to language
        # using above templates
        def sub_func(_):
            atom = matches.pop(0)
            if isinstance(atom ,tuple):
                return exp[1](*atom)
            else:
                return exp[1](atom)
        for exp in exprs:
            if uniform(0,1) < exp[2]:
                matches = re.findall(exp[0], formula)
                formula = re.sub(exp[0], sub_func, formula)

        # Post-processing
        if True:
            formula = formula.split(' + ')
            formula_tmp = [re.sub(r"\(([^\)]+)\)", r"\1", f) for f in formula]  # Remove brackets
            # Join clauses together with randomly selected connectives
            formula = formula_tmp[0]
            for f in formula_tmp[1:]:
                formula += f" {and_aux2()} {f} "
            formula = re.sub(r"\s+", " ", formula).strip() + '.'
            formula = re.sub(r" (,|\.) ", r"\1 ", formula)
            formula = re.sub(fr"\.?{sep}([a-zA-Z])", lambda m: fr". {sep} " + m.group(1).upper(), formula) # Fullstop after first sentence and uppercase first letter of second sentence.
            formula = re.sub(fr"\. ([a-z])", lambda m: fr". {m.group(1).upper()}", formula) # Capitalize after fullstop

        assert not re.search(r'_|\(|\)|\+', formula), (
            f"Haven't fully converted the template to language. Failed for: \n{_formula} \n{formula}")

        # print(_formula, '\n', __formula, '\n', formula, '\n\n')
        return formula

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

    @staticmethod
    def list_clauses(path: str):
        ''' List the different clauses in a prop logic dataset
        '''
        df = pd.read_csv(path).iloc[:15, :]
        formulae = df['sentence1'].tolist() + df['sentence2'].tolist()
        
        def func(s):
            s = s.replace('_', '')
            s = re.sub(r"^\((~?[a-z](?: [^\s]+ ~?[a-z])+)\)$", r"\1", s)
            return s

        formulae = [[func(term) for term in form.split(' & ')] for form in formulae]
        # unique = pd.Series(formulae).unique()
        [print(u) for u in formulae]
        

if __name__ == '__main__':
    path = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/logical-entailment-dataset'
    variant = 2
    PropositionalFormulaeProcessor().load_formulae(path, 'validate', variant).to_csv(
        os.path.join(path, f'v{variant}', 'dev.csv'))

    PropositionalFormulaeProcessor().load_formulae(path, 'train', variant).to_csv(
        os.path.join(path, f'v{variant}', 'train.csv'))

    # TemplateUtils.list_clauses(os.path.join(path, 'v2/dev.csv'))

    # def apply_str_sub(formula: list, exp: tuple):
    # ''' Helper to apply string subsitutions using regex.
    # '''
    # if re.search(exp[0], ' '.join(formula)):    # Skip if pattern not in string
    #     for n,_ in enumerate(formula):
    #         if re.search(exp[0], formula[n]):   # Check if pattern in term
    #             # Apply substitution to atoms in term
    #             for literals in re.findall(exp[0], formula[n]):
    #                 if isinstance(literals, tuple):
    #                     formula[n] = re.sub(exp[0], exp[1](*literals), formula[n], 1)
    #                 else:
    #                     formula[n] = re.sub(exp[0], exp[1](literals), formula[n], 1)
    # return formula
    # formulae = apply_str_sub(formula, exp)
