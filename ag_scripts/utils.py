import argparse
import json
import glob
import os

def dump_args(args: argparse.Namespace):
    with open('./ag_scripts/args/set_1.txt', 'w') as f:
        json.dump(vars(args), f, indent=2)

def load_args():
    with open('./ag_scripts/args/set_1.txt', 'r') as f:
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

if __name__ == '__main__':
    # parse_names_and_locs()
    pass