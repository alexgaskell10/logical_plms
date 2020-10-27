import sys
import os
import json
import csv
from random import choice

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


class Template:
    def __init__(self, dset):
        self.samples = []
        self.dset = dset
        self.load_names_locs()

    def load_names_locs(self):
        with open('data/names_locations.txt', 'r') as f:
            data = json.loads(f.read())

        self.locations = data['locations']
        self.names = data['names']


class NegationTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset)
        self.counter = counter
        self.set_labels()
        self.set_vars()
        self.pos_i()
        self.pos_ii()
        self.pos_iii()
        self.pos_iv()
        self.neg_i()
        self.neg_ii()
        self.neg_iii()
        self.neg_iv()

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
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+1}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+2}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+3}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_i(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+4}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+5}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+6}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+7}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })


class DisjunctionTemplate(Template):
    def __init__(self, dset, counter=0):
        super().__init__(dset)
        self.counter = counter
        self.set_labels()
        self.set_vars()
        self.pos_i()
        self.pos_ii()
        self.pos_iii()
        self.pos_iv()
        self.neg_i()
        self.neg_ii()
        self.neg_iii()
        self.neg_iv()

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
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+1}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_1}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+2}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def pos_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_1} or {self.location_2}."
        gold_label = self.pos_label
        pairID = f"{self.dset[0]}-{self.counter+3}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_i(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+4}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_ii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+5}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_iii(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_1}."
        sentence2 = f"{self.person_1} or {self.person_2} {self.pos_verb} {self.location_2}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+6}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })

    def neg_iv(self):
        sentence1 = f"{self.person_1} {self.pos_verb} {self.location_1}. {self.person_2} {self.pos_verb} {self.location_2}."
        sentence2 = f"{self.person_1} {self.pos_verb} {self.location_2} or {self.location_3}."
        gold_label = self.neg_label
        pairID = f"{self.dset[0]}-{self.counter+7}"
        captionID = f"c-{pairID}"
        self.samples.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_label": gold_label,
            "pairID": pairID,
            "captionID": captionID,
        })


class Namespace:
    def __init__(self, task):
        self.iters = 256
        self.dev_split = 0.25

        self.configure_outdir()
        self.set_task()

    def set_task(self, task):
        self.template = NegationTemplate

    def configure_outdir(self, task):
        assert task.lower() in ['negation', 'disjunction']
        self.outdir = os.path.join('data', task)
        # os.makekdirs(self.outdir, exist_ok=True)


if __name__ == '__main__':
    args = Namespace('negation')
    TemplateSampleGenerator(args)