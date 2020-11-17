import sys
import os
import json
import csv
import string
from random import choice
from utils import TemplateUtils
import pandas as pd

class TemplateSampleGenerator:
    def __init__(self, args):
        self.template = args.template
        self.iters = args.iters
        self.outdir = args.outdir
        self.dev_split = args.dev_split
        self.formulae_dir = args.formulae_dir
        self.task = args.task

        self.generate_samples()

    def generate_samples(self):
        ''' Generate all samples using the template and 
            save to file.
        '''
        # Create samples
        if self.template != PropositionalTemplate:
            train_samples = []
            for i in range(int(self.iters*(1-self.dev_split))):
                train_samples.extend(self.template('train').samples)

            dev_samples = []
            for i in range(int(self.iters*self.dev_split)):
                dev_samples.extend(self.template('dev').samples)
        else:
            self.formulae_dir = os.path.join(self.formulae_dir, self.task.strip('propositional_'))
            dev_samples = self.template('dev', self.formulae_dir).samples
            train_samples = self.template('train', self.formulae_dir).samples

        self.save_files(dev_samples, 'dev')
        self.save_files(train_samples, 'train')

    def save_files(self, samples, name):
        # Write to json
        with open(os.path.join(self.outdir, f'{name}.json'), 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')

        # Write to tsv
        keys = ['pairID', 'sentence1', 'sentence2', 'gold_label']
        with open(os.path.join(self.outdir, f'challenge_{name}.tsv'), 'w') as outfile:
            dict_writer = csv.DictWriter(outfile, keys, delimiter='\t')
            dict_writer.writerows(
                [{k:v for k,v in sample.items() if k in keys} for sample in samples])


class Template(TemplateUtils):
    ''' Base class for creating synthetic data using templates.
        Templates should inherit from this class and implement a
        number of methods to generate synthetic data in the child
        class.
    '''
    def __init__(self, dset: str, counter: int):
        super().__init__()
        self.samples = []
        self.dset = dset
        self.counter = counter
        self.load_names_locs()
        self.set_labels()

    def create_samples(self):
        for label in ['pos', 'neg']:
            for num in ['i', 'ii', 'iii', 'iv']:
                # Call child class methods
                s1, s2, pairID = getattr(self, f"{label}_{num}")()
                
                # Add distractor sentences and shuffle sentence order
                # to prevent the model from memorizing
                # s1 = self.add_distractors(s1)
                # s1 = Template.shuffle_sentence(s1)
                
                # Save relevant info
                gold_label = getattr(self, f"{label}_label")
                captionID = f"c-{pairID}"
                self.samples.append({
                    "sentence1": s1,
                    "sentence2": s2,
                    "gold_label": gold_label,
                    "pairID": pairID,
                    "captionID": captionID,
                })

    def load_names_locs(self):
        with open('data/names_locations.txt', 'r') as f:
            data = json.loads(f.read())

        self.locations = data['locations']
        self.names = data['names']

    def set_labels(self):
        raise NotImplementedError

    def set_vars(self):
        raise NotImplementedError


class PropositionalTemplate(Template):
    def __init__(self, dset, formulae_dir, counter=0):
        super().__init__(dset, counter)
        self.dset = dset
        self.formulae_dir = formulae_dir
        self.process_formulae()

    def set_labels(self):
        self.pos_verb = 'visited'
        self.neg_verb = 'did not visit'
        self.pos_label = "ENTAILMENT"
        self.neg_label = "CONTRADICTION"

    def process_formulae(self):
        ''' Load datasets of propositional logic formulae and convert them
            into natural language templates.
        '''
        df = pd.read_csv(os.path.join(self.formulae_dir, f"{self.dset}.csv")).iloc[70:100, :]

        # Convert propositional formulae to natural language here.
        # First join sentence1 & sentence2 (with seperator),
        # iterate through the formula & convert to natural language
        # using templates, then split column back into sentence1
        # & sentence2
        df['sentence'] = df[['sentence1', 'sentence2']].agg(lambda x: x[0].split() + ['§§§'] + x[1].split(), axis=1)
        # df['sentence'] = df['sentence'].apply(self.convert_to_template_v1)
        df['sentence'] = df['sentence'].apply(self.convert_to_template_v2)
        df[['sentence1', 'sentence2']] = df.sentence.str.split("§§§", expand=True).applymap(lambda x: x.strip())
        df.drop('sentence', axis=1, inplace=True)
        
        # Convert labels to correct terms
        df['gold_label'] = df.gold_label.map({0: self.neg_label, 1: self.pos_label})
        
        # Add pairID and captionID columns
        df['pairID'] = df.index.map(lambda x: f'{self.dset[0]}-{str(x)}')
        df['captionID'] = df.pairID.map(lambda x: f'c-{x}')      

        # Drop rows with tokens > bert's limit
        len_flag = (df['sentence1'].str.split().apply(len) + df['sentence2'].str.split().apply(len)) <= 400
        df = df[len_flag]

        self.samples = df.to_dict('records')


class NegationTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)
        self.set_vars()
        self.create_samples()

    def set_labels(self):
        self.pos_verb = 'visited'
        self.neg_verb = 'did not visit'
        self.pos_label = "ENTAILMENT"
        self.neg_label = "CONTRADICTION"

    def set_vars(self):
        self.person_1 = self.names.pop(choice(range(len(self.names))))
        self.person_2 = self.names.pop(choice(range(len(self.names))))
        self.location_1 = self.locations.pop(choice(range(len(self.locations))))
        self.location_2 = self.locations.pop(choice(range(len(self.locations))))
        self.location_3 = self.locations.pop(choice(range(len(self.locations))))

        if False:
            print(self.person_1, self.person_2)
            print(self.location_1, self.location_2, self.location_3)

    def pos_i(self):
        ''' p |= -q '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter}"
        return sentence1, sentence2, pairID

    def pos_ii(self):
        ''' p, q |= p '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+1}"
        return sentence1, sentence2, pairID

    def pos_iii(self):
        ''' p, -q |= p '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+2}"
        return sentence1, sentence2, pairID

    def pos_iv(self):
        ''' p, -q |= -q '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_2} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+3}"
        return sentence1, sentence2, pairID

    def neg_i(self):
        ''' -p |= p '''
        sentence1 = f"{self.person_1} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+4}"
        return sentence1, sentence2, pairID

    def neg_ii(self):
        ''' p, -q |= -p '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+5}"
        return sentence1, sentence2, pairID

    def neg_iii(self):
        ''' p, -q |= q '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+6}"
        return sentence1, sentence2, pairID

    def neg_iv(self):
        ''' p, q |= -q '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+7}"
        return sentence1, sentence2, pairID


class NegationDisjunctionTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)
        self.set_vars()
        self.create_samples()

    def set_labels(self):
        self.pos_verb = 'visited'
        self.neg_verb = 'did not visit'
        self.pos_label = "ENTAILMENT"
        self.neg_label = "CONTRADICTION"

    def set_vars(self):
        self.person_1 = self.names.pop(choice(range(len(self.names))))
        self.person_2 = self.names.pop(choice(range(len(self.names))))
        self.location_1 = self.locations.pop(choice(range(len(self.locations))))
        self.location_2 = self.locations.pop(choice(range(len(self.locations))))
        self.location_3 = self.locations.pop(choice(range(len(self.locations))))

        if False:
            print(self.person_1, self.person_2)
            print(self.location_1, self.location_2, self.location_3)

    def pos_i(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2} or {self.location_3}."
        pairID = f"{self.dset[0]}-{self.counter}"
        return sentence1, sentence2, pairID

    def pos_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+1}"
        return sentence1, sentence2, pairID

    def pos_iii(self):
        sentence1 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+2}"
        return sentence1, sentence2, pairID

    def pos_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+3}"
        return sentence1, sentence2, pairID

    def neg_i(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_1} or {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+4}"
        return sentence1, sentence2, pairID

    def neg_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}. {self.person_1} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+5}"
        return sentence1, sentence2, pairID

    def neg_iii(self):
        sentence1 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+6}"
        return sentence1, sentence2, pairID

    def neg_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+7}"
        return sentence1, sentence2, pairID


class DisjunctionTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)
        self.set_vars()
        self.create_samples()

    def set_labels(self):
        self.pos_verb = 'visited'
        self.neg_verb = 'did not visit'
        self.pos_label = "ENTAILMENT"
        self.neg_label = "CONTRADICTION"

    def set_vars(self):
        self.person_1 = self.names.pop(choice(range(len(self.names))))
        self.person_2 = self.names.pop(choice(range(len(self.names))))
        self.location_1 = self.locations.pop(choice(range(len(self.locations))))
        self.location_2 = self.locations.pop(choice(range(len(self.locations))))
        self.location_3 = self.locations.pop(choice(range(len(self.locations))))

        if False:
            print(self.person_1, self.person_2)
            print(self.location_1, self.location_2, self.location_3)

    def pos_i(self):
        ''' p |= p v q '''
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter}"
        return sentence1, sentence2, pairID

    def pos_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+1}"
        return sentence1, sentence2, pairID

    def pos_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+2}"
        return sentence1, sentence2, pairID

    def pos_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+3}"
        return sentence1, sentence2, pairID

    def neg_i(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+4}"
        return sentence1, sentence2, pairID

    def neg_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        pairID = f"{self.dset[0]}-{self.counter+5}"
        return sentence1, sentence2, pairID

    def neg_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+6}"
        return sentence1, sentence2, pairID

    def neg_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        pairID = f"{self.dset[0]}-{self.counter+7}"
        return sentence1, sentence2, pairID


class Namespace:
    def __init__(self, task):
        self.iters = 2048
        self.dev_split = 0.25
        self.formulae_dir = '/vol/bitbucket/aeg19/logical_plms/semantic_fragments/ag_scripts/data/logical-entailment-dataset'
        self.task = task

        self.configure_outdir()
        self.set_task()

    def set_task(self):
        if self.task.lower() == 'negation':
            self.template = NegationTemplate
        elif self.task.lower() == 'disjunction':
            self.template = DisjunctionTemplate
        elif self.task.lower() == 'negation_disjunction':
            self.template = NegationDisjunctionTemplate
        elif 'propositional' in self.task.lower():
            self.template = PropositionalTemplate

    def configure_outdir(self):
        # assert self.task.lower() in ['negation', 'disjunction', 'negation_disjunction', 'propositional', 'propositional_v2']
        self.outdir = os.path.join('data', self.task)
        os.makedirs(self.outdir, exist_ok=True)


if __name__ == '__main__':
    # for x in ['negation', 'disjunction', 'negation_disjunction']:
    for x in ['propositional_v3']:
        args = Namespace(x)
        TemplateSampleGenerator(args)
