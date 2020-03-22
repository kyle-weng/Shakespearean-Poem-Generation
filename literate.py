"""some parsing and processing functions"""

from collections import defaultdict

def futz(wdict, word):
    """attempt to match words to ones in dictionary"""
    if word in wdict:
        return word
    # try removing quotes
    unquoted = word.strip("'")
    if unquoted in wdict:
        return unquoted

def read(fname, wdict=None, bad=",.:;()?!"):
    """Returns list of sonnets, and set of words in a file.
    
    Each sonnet is a list of lines; each line is a list of words
    """
    drops = {ord(c):None for c in bad}
    with open(fname, "r") as f:
        lines = [l.strip().lower().translate(drops).split() for l in f]

    data = []
    words = set()
    for l in lines:
        if (len(l) == 0): # drop blank lines
            continue
        if (len(l) == 1): # assume single word line delimits new sonnet
            data.append([])
            continue
        if (wdict):
            l = [futz(wdict, w) for w in l]
        # append to last sonnet in list
        data[-1].append(l)
        words.update(l)

    return data, words

def recite(sonnet):
    """Assemble sonnet back into string and print it"""
    print("\n".join([" ".join(l) for l in sonnet]))

def syldict(fname):
    """Reads syllable dictionary,
    and returns a dictionary mapping words
    to a list of possible numbers of syllables.
    Negative numbers are used to indicate ending only counts
    (see syllable dictionary explanation)
    """
    # replace E with negative sign
    trans = {ord("E"):"-"}
    with open(fname, "r") as f:
        lines = [l.translate(trans).split() for l in f]

    d = {}
    for l in lines:
        d[l[0]] = [int(c) for c in l[1:]]
    return d

def get_rhymes(sonnets):
    """Returns dictionary mapping words to list of rhyming words,
    based on their use in sonnets"""
    rhymes = defaultdict(set)
    # lines which should rhyme
    pairs = [(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,13)]
    for s in sonnets:
        if (len(s) != 14):
            continue
        for i,j in pairs:
            a,b = s[i][-1], s[j][-1]
            rhymes[a].add(b)
            rhymes[b].add(a)
    return rhymes