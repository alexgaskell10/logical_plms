import re, sys, glob

def translate(fn,n=0):
    st = ""
    if n==0: 
        n = len(fn)
    for fname in fn[0:n]:
        with open(fname) as fh:
            s = fh.read()
        # print(s)
        if len(s)<5000:
            prb = re.search('% *Problem *\: *(.*?)\n',s)
            if prb:
                prb = prb.group(1)
            else:
                prb = re.search('% *Axioms *\: *(.*?)\n',s)
                prb = prb.group(1) if prb else ""
            
            dsc = re.search('% *English *\: *(.*?)% Refs',s,flags=re.S)
            dsc = dsc.group(1) if dsc else ""
            dsc = re.sub(r'% +', r'', dsc)
            pat = re.compile('^( *?)%(.*?)\n',re.M)
            s = pat.sub('',s) # remove comments
            pat = re.compile('\/\*(.*?)\*\/',re.S)
            s = pat.sub('',s) # remove long comments
            return parse(s)

def parse(s):
    ''' Convert TPTP format to problog-friendly format '''
    if 'cnf(' in s:
        return parse_cnf(s)
    else:
        raise NotImplementedError(f'Not implemented for {s}')

def parse_cnf(s):
    ''' Convert TPTP CNF problem to problog-friendly format
            - s: CNF problem string
    '''
    # Convert to clausal form
    s = re.sub(r' +',' ',s)
    s = re.sub(r'\. *?\n','.n.',s)
    s = re.sub(r'cnf\(.*\n','',s)
    s = re.sub(r'\n|\( | \)\)| \)|\)\)','',s)
    # print(s); sys.exit()

    # Make problog-friendly:
    #   - Facts of form: polarity (predicate v1 v2 ...) 
    #       e.g. - (eats X 'anne')
    #   - Rules of form: [ polarity clause , polarity clause , ... ] -> polarity clause
    #       e.g. [ - (red X) , - (eats X anne) ] -> - (eats anne charlie)
    s = re.sub(r'(?<=\()([a-z_]+)', r"'\1'", s)      # Wrap constants within ''
    s = re.sub(r'(?<![A-Za-z_])([a-z_]+)(?=\))', r"'\1'", s)       # Wrap constants within ''
    s = re.sub('~', '-', s)     # Prepend negative terms with "-"
    s = re.sub(r'(\.n\.\s|\s\s|\|\s)([a-z])', r'\1+ \2', ' '+s)     # Prepend positive terms with "+"
    s = re.sub(r' ([a-z_]+)\(', r' (\1 ', s)
    s = s.replace(',', ' ')
    # Convert rules
    s = re.sub(r'\| \+ (\([a-zA-Z_ \']+\))(?=\.n\.)', r'-> - \1', s)
    s = re.sub(r'\| \- (\([a-zA-Z_ \']+\))(?=\.n\.)', r'-> + \1', s)
    s = s.replace(' | ',' , ')
    s = '.n. '.join([re.sub(r'(.+)(?=->)', r'[ \1 ] ', x) for x in s.strip().strip('.n.').split('.n. ')])
    s = enforce_valid_theory(s)
    # print(s); sys.exit()
    
    rules, facts = [], []
    for f in s.split('.n. '):
        rules.append(f) if '->' in f else facts.append(f)
    # print('\n'.join(rules), '\n', '\n'.join(facts)); sys.exit()
    return rules, facts

def enforce_valid_theory(s):
    ''' Ensure all predicates are defined in the theory.
            - s: CNF theory in problog-friendly format
    '''
    fs = s.split('.n. ')
    counter = 0
    preds = list(set(re.findall(r'(?<= \()[a-z_]+(?=[ \)])', s)))
    
    # Check if all predicates have been defined. Return if so,
    # otherwise re-define formulae so all are defined
    while True:    
        # Compute number of times each predicate is defined
        res = {}
        for pred in preds:
            matches = len(re.findall(rf'\({pred} [a-zA-Z _\']+\).n.', s))
            res[pred] = matches
        # res = [re.search(rf'\({pred} [a-zA-Z _\']+\).n.', s) for pred in preds]
        # Proceed if so all have been defined at least once
        if all(res.values()):
            return '.n. '.join(fs)
        # Rearrage relevant clauses so that undefined predicate is defined
        undef = [k for k,v in res.items() if v == 0].pop()
        fs = s.split('.n. ')
        for f in fs:
            if re.search(rf'(?<= \(){undef}(?=[ \)])', f) and res[f] > 0:
                print(f); sys.exit()
    print(s); sys.exit()

def process(domain, tptp_path):
    fn = sorted(glob.glob(tptp_path+"Problems/"+domain+"/*-*.p"))[:1]
    print(fn)
    rules, facts = translate(fn)
    return rules, facts

tptp_path = "/vol/bitbucket/aeg19/logical_plms/data/TPTP-v7.4.0/"
doms = ["PUZ",]

for d in doms:
    rules, facts = process(d, tptp_path)


    from theory_generator import run_theory_in_problog
    from utils import *
    from common import *

    facts = [parse_fact(f) for f in facts]
    rules = [parse_statement(r) for r in rules]
    theory = Theory(facts, rules)
    # print(rules); sys.exit()
    # # theory = Theory(None, facts)
    # assertion_statement = "+ (killed X 'agatha')"
    # assertion = parse_fact(assertion_statement)

    # # run_theory_in_problog(theory, assertion)

    # import problog
    # from problog.program import PrologString
    # from problog.core import ProbLog
    # from problog import get_evaluatable
    # from problog.engine import NonGroundProbabilisticClause, UnknownClause
    # from problog.engine_stack import NegativeCycle
    # from problog.formula import LogicFormula, LogicDAG
    # from problog.sdd_formula import SDD

    # program = theory.program("problog", assertion)
    # # program = "1.0::red('bald_eagle').\nquery(red('tiger'))."
    # program = "1.0::lives('agatha').\n1.0::lives('butler').\n1.0::lives('charles').\n1.0::hates('agatha', 'agatha').\n1.0::hates('agatha', 'charles').\n1.0::richer(X, Y) :- \\+killed(X, Y).\n1.0::hates('charles', X) :- \\+hates('agatha', X).\n1.0::hates(X, 'charles') :- \\+hates(X, 'agatha'), \\+hates(X, 'butler').\n1.0::killed(X, Y) :- hates(X, Y).\n1.0::\\+hates('butler', X) :- \\+hates('agatha', X).\n1.0::\\+hates('butler', X) :- \\+lives(X), richer(X, 'agatha').\n1.0::\\+killed('charles', 'agatha') :- killed('butler', 'agatha').\nquery(killed('agatha', 'agatha'))."
    # print(program)
    # lf = LogicFormula.create_from(program)
    # print('done')

'''
1.0::\+sees('bear', 'squirrel').
1.0::\+sees('bald_eagle', 'cat').
1.0::kind('dog').
1.0::\+red('rabbit').
1.0::visits('dog', 'squirrel').
1.0::red('bald_eagle').
1.0::\+visits('squirrel', 'tiger').
1.0::chases('mouse', 'lion').
1.0::\+likes('tiger', 'dog').
1.0::\+nice(X) :- \+likes(X, 'rabbit'), \+green(X).
1.0::\+red(X) :- \+eats(X, 'dog').
1.0::\+blue(X) :- \+sees(X, 'dog').
1.0::\+kind(X) :- visits(X, 'mouse').
1.0::blue(X) :- \+big(X).
query(red('tiger')).
'''

'''
1.0::lives('agatha').
1.0::lives('butler').
1.0::lives('charles').
1.0::hates('agatha', 'agatha').
1.0::hates('agatha', 'charles').
1.0::killed(X, Y) :- \+richer(X, Y).
1.0::richer(X, Y) :- \+killed(X, Y).
1.0::hates('charles', X) :- \+hates('agatha', X).
1.0::hates(X, 'charles') :- \+hates(X, 'agatha'), \+hates(X, 'butler').
1.0::\+hates(X, Y) :- \+killed(X, Y).
1.0::\+hates('butler', X) :- \+hates('agatha', X).
1.0::\+hates('butler', X) :- \+lives(X), richer(X, 'agatha').
1.0::\+killed('charles', 'agatha') :- killed('butler', 'agatha').
query(killed('agatha', 'agatha')).
'''

"1.0::lives('agatha').\n1.0::lives('butler').\n1.0::lives('charles').\n1.0::hates('agatha', 'agatha').\n1.0::hates('agatha', 'charles').\n1.0::killed(X, Y) :- \\+richer(X, Y).\n1.0::richer(X, Y) :- \\+killed(X, Y).\n1.0::hates('charles', X) :- \\+hates('agatha', X).\n1.0::hates(X, 'charles') :- \\+hates(X, 'agatha'), \\+hates(X, 'butler').\n1.0::\\+hates(X, Y) :- \\+killed(X, Y).\n1.0::\\+hates('butler', X) :- \\+hates('agatha', X).\n1.0::\\+hates('butler', X) :- \\+lives(X), richer(X, 'agatha').\n1.0::\\+killed('charles', 'agatha') :- killed('butler', 'agatha').\nquery(killed('agatha', 'agatha'))."

"1.0::\\+sees('bear', 'squirrel').\n1.0::\\+sees('bald_eagle', 'cat').\n1.0::kind('dog').\n1.0::\\+red('rabbit').\n1.0::visits('dog', 'squirrel').\n1.0::red('bald_eagle').\n1.0::\\+visits('squirrel', 'tiger').\n1.0::chases('mouse', 'lion').\n1.0::\\+likes('tiger', 'dog').\n1.0::\\+nice(X) :- \\+likes(X, 'rabbit'), \\+green(X).\n1.0::\\+red(X) :- \\+eats(X, 'dog').\n1.0::\\+blue(X) :- \\+sees(X, 'dog').\n1.0::\\+kind(X) :- visits(X, 'mouse').\n1.0::blue(X) :- \\+big(X).\nquery(red('tiger'))."