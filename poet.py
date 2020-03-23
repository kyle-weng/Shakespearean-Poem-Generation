"""Functions for generating sonnets using HMM
and visualizing HMMs"""

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def sonnetify(wmap, syldict, ids):
    """take string of word ids,
    convert them to words,
    and break lines every 10 ish syllables
    """
    sonnet = []
    line = []
    syls = 0
    for i in ids:
        word = wmap.inv[i]
        syls += abs(syldict[word][0])
        line.append(word)
        if (syls >= 10):
            sonnet.append(line)
            line = []
            syls = 0
            if (len(sonnet) == 14):
                break
    return sonnet

def wordmask(wmap, condition):
    """creates a len(words) array that is True
    for words satisfing the condition
    """
    mask = np.repeat(False, len(wmap))
    filtered = [wmap[w] for w in wmap if condition(w)]
    mask[filtered] = True
    return mask

def generate_sonnet(hmm, wmap, rhymes, syldict):
    '''
    Generates sonnet using hmm

    Returns:
        emission:   The randomly generated sonnet as a list of words
    '''
    state = np.random.choice(hmm.L, p=hmm.A0)
    
    # functions to filter down allowable words
    def fewersyls(n, w):
        return abs(syldict[w][0]) < n
    def hassyls(n, w):
        return abs(syldict[w][0]) == n
    def rhymable(w):
        return len(rhymes[w]) > 0
    def rhymeswith(word, w):
        return w in rhymes[word]
    
    def try_line(state, rhymeword=None, maxwords=20):
        """Generate new line starting with given state,
        having 10 syllables, ending on a rhymable word if rhymeword is None,
        and ending on a rhyme with rhymeword if it is a string.
        It unfortunately often gets stuck generating th'.
        """
        syls = 10
        line = []
        while (syls > 0):
            if (len(line) > maxwords):
                break
            # if a word either has few enough syllables to not end the line,
            # or ends the line and satisfies the rhyme scheme.
            # If there is no rhymeword, it must be rhymable, otherwise,
            # it must rhyme with rhymeword.
            sylrhyme = lambda w: fewersyls(syls, w)\
                or (hassyls(syls, w) and (rhymeswith(rhymeword, w) if rhymeword else rhymable(w)))
            
            # filter out words by setting their probabilities to zero
            pobs = hmm.O[state].copy()
            pobs[~wordmask(wmap, sylrhyme)] = 0
            pobs /= pobs.sum()

            # pick new word and state
            word = wmap.inv[np.random.choice(hmm.D, p=pobs)]
            state = np.random.choice(hmm.L, p=hmm.A[state])
            line.append(word)
            syls -= abs(syldict[word][0])
            #print(syls, word, rhymeword, sum(pobs!=0))
        return state, line
    
    def generate_line(state0, rhymeword=None, maxwords=20, tries=100):
        """Keeps re-running try_line until it generates something valid"""
        for _ in range(tries):
            state, line = try_line(state0, rhymeword, maxwords)
            if (len(line) < maxwords):
                return state,line
        # we tried our best
        return state,line

    sonnet = []
    # make three quadtrains, with rhyme scheme abab
    for q in range(3):
        state, line = generate_line(state)
        sonnet.append(line)
        state, line = generate_line(state)
        sonnet.append(line)
        state, line = generate_line(state, sonnet[-2][-1])
        sonnet.append(line)
        state, line = generate_line(state, sonnet[-2][-1])
        sonnet.append(line)
    # and a rhyming couplet
    state, line = generate_line(state)
    sonnet.append(line)
    state, line = generate_line(state, sonnet[-1][-1])
    sonnet.append(line)

    return sonnet

"""Visualization functions,
heavily based on assignment 6
"""
def circle(r=128):
    """Boolean mask of circle with specified radius."""
    # grid containing numbers [-r, r]
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x**2 + y**2 <= r**2)

def state_wordcloud(wmap, pobs, r=128):
    """returns wordcloud generated from probabilities array,
    with the reasonable-looking defaults from the assignment"""
    freqs = {word: pobs[wmap[word]] for word in wmap}
    mask = 255 * (1 - circle(r=r)) # wordcloud draws in black area
    return WordCloud(max_words=50,background_color="white",mask=mask)\
        .generate_from_frequencies(freqs)

def visualize_transitions(wmap, hmm, R=512, r=128):
    """Plots wordclouds for each state in circle,
    and draws arrows for transition probabilities
    """
    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(R/50)
    fig.set_figwidth(R/50)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, 2*(R+r)])
    ax.set_ylim([0, 2*(R+r)])
    n_states = hmm.L

    # precompute positions of all states on circle
    spos = []
    for i in range(n_states):
        direction = np.array([
            np.cos(np.pi * 2 * i / n_states),
            np.sin(np.pi * 2 * i / n_states)
        ])
        spos.append(R + r + R*direction)

    # draw wordclouds
    for i in range(n_states):
        wordcloud = state_wordcloud(wmap, hmm.O[i], r=r)
        x,y = spos[i]
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # draw arrows
    for i in range(n_states):
        for j in range(n_states):
            x,y = spos[i] # start at center of state wordcloud
            direction = spos[j] - spos[i] # draw towards target state
            # but don't go into it
            dist = np.sqrt(np.sum(np.square(direction)))
            dx,dy = (0,0) if dist < 1 else direction * (dist - r) / dist
            # this is indeed getting pretty jank pretty fast
            ax.arrow(x,y, dx,dy,
                color=(0,0,0,hmm.A[i][j]),
                head_width=R/32
            )
    
    return ax
