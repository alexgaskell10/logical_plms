import sys
import os
import json
import csv
from random import choice
from utils import TemplateUtils

class TemplateSampleGenerator:
    def __init__(self, args):
        self.template = args.template
        self.iters = args.iters
        self.outdir = args.outdir
        self.dev_split = args.dev_split

        self.generate_samples()

    def generate_samples(self):
        ''' Generate all samples using the template and 
            save to file.
        '''
        # Create samples
        train_samples = []
        for i in range(int(self.iters*(1-self.dev_split))):
            train_samples.extend(self.template('train').samples)

        dev_samples = []
        for i in range(int(self.iters*self.dev_split)):
            dev_samples.extend(self.template('dev').samples)

        # Write to json
        with open(os.path.join(self.outdir, 'train.json'), 'w') as f:
            for sample in train_samples:
                json.dump(sample, f)
                f.write('\n')

        with open(os.path.join(self.outdir, 'dev.json'), 'w') as f:
            for sample in dev_samples:
                json.dump(sample, f)
                f.write('\n')

        # Write to tsv
        keys = ['pairID', 'sentence1', 'sentence2', 'gold_label']
        with open(os.path.join(self.outdir, 'challenge_train.tsv'), 'w') as outfile:
            dict_writer = csv.DictWriter(outfile, keys, delimiter='\t')
            dict_writer.writerows(
                [{k:v for k,v in sample.items() if k in keys} for sample in train_samples])

        with open(os.path.join(self.outdir, 'challenge_dev.tsv'), 'w') as outfile:
            dict_writer = csv.DictWriter(outfile, keys, delimiter='\t')
            dict_writer.writerows(
                [{k:v for k,v in sample.items() if k in keys} for sample in dev_samples])


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
        self.set_vars()
        self.create_samples()

    def load_names_locs(self):
        with open('data/names_locations.txt', 'r') as f:
            data = json.loads(f.read())

        self.locations = data['locations']
        self.names = data['names']

    def set_labels(self):
        raise NotImplementedError

    def set_vars(self):
        raise NotImplementedError

    def create_samples(self):
        for label in ['pos', 'neg']:
            for num in ['i', 'ii', 'iii', 'iv']:
                # Call child class methods
                s1, s2, pairID = getattr(self, f"{label}_{num}")()
                
                # Add distractor sentences and shuffle sentence order
                # to prevent the model from memorizing
                s1 = self.add_distractors(s1)
                s1 = Template.shuffle_sentence(s1)
                
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


class NegationDisjunctionTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)

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


class NegationTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)

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
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter}"
        return sentence1, sentence2, pairID

    def pos_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+1}"
        return sentence1, sentence2, pairID

    def pos_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+2}"
        return sentence1, sentence2, pairID

    def pos_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_2} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+3}"
        return sentence1, sentence2, pairID

    def neg_i(self):
        sentence1 = f"{self.person_1} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+4}"
        return sentence1, sentence2, pairID

    def neg_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.neg_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_1}."
        pairID = f"{self.dset[0]}-{self.counter+5}"
        return sentence1, sentence2, pairID

    def neg_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.neg_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+6}"
        return sentence1, sentence2, pairID

    def neg_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_1} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.neg_verb} {self.location_2}."
        pairID = f"{self.dset[0]}-{self.counter+7}"
        return sentence1, sentence2, pairID


class DisjunctionTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset, counter)

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

    def configure_outdir(self):
        assert self.task.lower() in ['negation', 'disjunction', 'negation_disjunction']
        self.outdir = os.path.join('data', self.task)
        os.makedirs(self.outdir, exist_ok=True)


if __name__ == '__main__':
    for x in ['negation', 'disjunction', 'negation_disjunction']:
        args = Namespace(x)
        TemplateSampleGenerator(args)