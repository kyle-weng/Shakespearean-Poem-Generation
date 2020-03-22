"""HMM implementation, adapated from assignment 6"""

import numpy as np
from collections import namedtuple
import time

Hmm = namedtuple("Hmm", ["A0", "A", "O", "D", "L"])

def init(A, O):
    A, O = np.array(A), np.array(O)
    L, D = O.shape
    return Hmm(np.ones(L)/L, A, O, D, L)

def init_rand(L, D):
    A = np.random.rand(L,L)
    A /= A.sum(axis=1).reshape((-1,1))
    O = np.random.rand(L,D)
    O /= O.sum(axis=1).reshape((-1,1))
    return init(A, O)

def viterbi(hmm, x):
    '''
    Uses the Viterbi algorithm to find the max probability state 
    sequence corresponding to a given input sequence.

    Arguments:
        x:          Input sequence in the form of a list of length M,
                    consisting of integers ranging from 0 to D - 1.

    Returns:
        max_seq:    State sequence corresponding to x with the highest
                    probability.
    '''

    A = np.log(hmm.A)
    O = np.log(hmm.O)
    M = len(x)      # Length of sequence.

    # The (i, j)^th elements of probs and steps are the max probability
    # of the prefix of length i ending in state j and the last state in the prefix
    # that gives this probability, respectively.
    #
    # For instance, probs[1][0] is the probability of the prefix of
    # length 1 ending in state 0.
    probs = np.ones((M, hmm.L))
    steps = -np.ones((M, hmm.L))


    probs[0] = np.log(hmm.A0) + O[:,x[0]]
    steps[0] = range(hmm.L)
    # for every element of x
    for i in range(1,M):
        # for every possible next state
        for j in range(hmm.L):
            # probability of going into state j and seeing x[i],
            # from each possible previous state
            poss = probs[i-1] + A[:,j] + O[j,x[i]]
            best = np.argmax(poss)
            probs[i,j] = poss[best]
            steps[i,j] = best

    # backtrack to recover sequence
    seq = np.zeros(M)
    state = np.argmax(probs[-1])
    for i in range(M):
        seq[-1-i] = state
        state = steps[-i-1,int(state)]

    return seq



def forward(hmm, x, normalize=False, log=False):
    '''
    Uses the forward algorithm to calculate the alpha probability
    vectors corresponding to a given input sequence.

    Arguments:
        x:          Input sequence in the form of a list of length M,
                    consisting of integers ranging from 0 to D - 1.

        normalize:  Whether to normalize each set of alpha_j(i) vectors
                    at each i. This is useful to avoid underflow in
                    unsupervised learning.

    Returns:
        alphas:     Vector of alphas.

                    The (i, j)^th element of alphas is alpha_j(i),
                    i.e. the probability of observing prefix x^1:i
                    and state y^i = j.

                    e.g. alphas[1][0] corresponds to the probability
                    of observing x^1:1, i.e. the first observation,
                    given that y^1 = 0, i.e. the first state is 0.
    '''

    M = len(x)
    O = hmm.O
    A = hmm.A

    alphas = -np.ones((M,hmm.L))
    facs = np.zeros(M) # log of multiplicative factor for normalization

    alphas[0] = hmm.A0 * O[:,x[0]]
    norm = np.sum(alphas[0])
    alphas[0] /= norm
    facs[0] = np.log(norm)

    # for every element of x
    for i in range(1,M):
        # for every possible next state
        for z in range(hmm.L):
            alphas[i,z] = O[z,x[i]] * np.sum(alphas[i-1] * A[:,z])
        norm = np.sum(alphas[i])
        alphas[i] /= norm
        facs[i] = facs[i-1]+np.log(norm)

    if (log):
        if (normalize):
            return np.log(alphas)
        return np.log(alphas) + facs.reshape((-1,1))
    if (normalize):
        return alphas
    return alphas * np.exp(facs).reshape((-1,1))


def backward(hmm, x, normalize=False):
    '''
    Uses the backward algorithm to calculate the beta probability
    vectors corresponding to a given input sequence.

    Arguments:
        x:          Input sequence in the form of a list of length M,
                    consisting of integers ranging from 0 to D - 1.

        normalize:  Whether to normalize each set of alpha_j(i) vectors
                    at each i. This is useful to avoid underflow in
                    unsupervised learning.

    Returns:
        betas:      Vector of betas.

                    The (i, j)^th element of betas is beta_j(i), i.e.
                    the probability of observing prefix x^(i+1):M and
                    state y^i = j.

                    e.g. betas[M][0] corresponds to the probability
                    of observing x^M+1:M, i.e. no observations,
                    given that y^M = 0, i.e. the last state is 0.
    '''

    M = len(x)      # Length of sequence.
    O = hmm.O
    A = hmm.A

    betas = -np.ones((M,hmm.L))
    betas[-1] = 1
    if (normalize):
        betas[-1] /= np.sum(betas[-1])
    # for every element of x
    for i in range(1,M):
        # for every possible next state
        for z in range(hmm.L):
            betas[-i-1,z] =  np.sum(betas[-i] * A[z,:]* O[:,x[-i]])
        if (normalize):
            betas[-i-1] /= np.sum(betas[-i-1])

    return betas


def supervised_learning(hmm, X, Y):
    '''
    Trains the HMM using the Maximum Likelihood closed form solutions
    for the transition and observation matrices on a labeled
    datset (X, Y).

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of
                    lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of
                    lists.

                    Note that the elements in X line up with those in Y.
    '''

    # Calculate each element of A using the M-step formulas.
    A_num = np.zeros((hmm.L,hmm.L))
    A_den = np.zeros_like(A_num)

    O_num = np.zeros((hmm.L,hmm.D))
    O_den = np.zeros_like(O_num)

    for x,y in zip(X, Y):
        for i in range(len(x)-1):
            # count probability of state transition
            A_num[y[i],y[i+1]] += 1
            A_den[y[i],:] += 1

            # count probability of observation
            O_num[y[i],x[i]] += 1
            O_den[y[i],:] += 1
        O_num[y[-1],x[-1]] += 1
        O_den[y[-1],:] += 1

    return hmm._replace(A=(A_num / A_den), O=(O_num / O_den))

def unsupervised_step(hmm, X):
    '''
    Single step of the Baum-Welch algorithm on an unlabeled
    datset X.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of length M, consisting of integers ranging
                    from 0 to D - 1. In other words, a list of lists.

        N_iters:    The number of iterations to train on.
    '''
    A_num = np.zeros((hmm.L,hmm.L))
    A_den = np.zeros(hmm.L)
    O_num = np.zeros((hmm.L,hmm.D))
    O_den = np.zeros(hmm.L)

    # for every sentence in training data
    for x in X:
        alphas = forward(hmm, x, True)
        betas = backward(hmm, x, True)

        # numerators
        Pa = alphas * betas
        Pa /= Pa.sum(axis=1).reshape((-1,1))

        # denominators
        den = Pa[:-1].sum(axis=0)
        A_den += den # A stops before end of sentence
        O_den += den + Pa[-1] # O goes to end of sentence
        for i in range(len(x)-1):
            # probability of state transition
            Pab = np.outer(alphas[i], betas[i+1]) * hmm.A * hmm.O[:,x[i+1]]
            A_num += Pab / np.sum(Pab)
            # probability of observation
            O_num[:,x[i]] += Pa[i]
        O_num[:,x[-1]] += Pa[-1]

    return hmm._replace(
        A = A_num / A_den.reshape((-1,1)),
        O = O_num / O_den.reshape((-1,1))
    )

def unsupervised_starts(hmm, X):
    """Weight starting state probabilities according to
    probabilities of observing initial observation on each sequence...
    Does this make theoretical sense?
    """
    A0 = np.zeros(hmm.L)
    for x in X:
        likes = hmm.O[:,x[0]]
        A0 += likes / likes.sum() / len(X)
    return hmm._replace(A0=A0)

def generate_emission(hmm, M):
    '''
    Generates an emission of length M, assuming that the starting state
    is chosen uniformly at random. 

    Arguments:
        M:          Length of the emission to generate.

    Returns:
        emission:   The randomly generated emission as a list.

        states:     The randomly generated states as a list.
    '''

    # random state, given current state
    rstate = lambda s: np.random.choice(hmm.L, p=hmm.A[s])
    # random observtion, given current state
    robs = lambda s: np.random.choice(hmm.D, p=hmm.O[s])

    # initialize
    states = [np.random.choice(hmm.L, p=hmm.A0)]
    emission = [robs(states[0])]
    for i in range(1,M):
        states.append(rstate(states[i-1]))
        emission.append(robs(states[i]))

    return emission, states


def probability_alphas(hmm, x, log=False):
    '''
    Finds the maximum probability of a given input sequence using
    the forward algorithm.

    Arguments:
        x:          Input sequence in the form of a list of length M,
                    consisting of integers ranging from 0 to D - 1.

    Returns:
        prob:       Total probability that x can occur.
    '''

    # Calculate alpha vectors.
    alphas = forward(hmm, x, normalize=False, log=log)

    # alpha_j(M) gives the probability that the state sequence ends
    # in j. Summing this value over all possible states j gives the
    # total probability of x paired with any state sequence, i.e.
    # the probability of x.
    if (log):
        m = np.max(alphas[-1])
        return np.log(np.sum(np.exp(alphas[-1]-m))) + m
    return sum(alphas[-1])

def score_ll(hmm, X):
    score = 0
    for x in X:
        score += probability_alphas(hmm, x, log=True) / len(x) / len(X)
    return score


def supervised(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    A = [[0 for i in range(L)] for j in range(L)]

    O = [[0 for i in range(D)] for j in range(L)]

    # Train an HMM with labeled data.
    HMM = init(A, O)
    return supervised_learning(HMM, X, Y)

def unsupervised(X, n_states=10, N_iters=50, hmm=None):
    '''
    Helper function to train an unsupervised HMM by running N_iters
    of unsupervised_step.
    If hmm is not provided, the function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, and creates the HMM.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
        
        hmm:        Hmm, assumed to have correct number of states/observations.
                    Useful for continuing training
    '''

    if (hmm is None):
        # Make a set of observations.
        observations = set()
        for x in X:
            observations |= set(x)

        # Compute L and D.
        L = n_states
        D = len(observations)

        hmm = init_rand(L, D)
    scores = np.zeros(N_iters+1)
    
    print("Training", N_iters, "iters")
    start = time.time()
    # epochs
    for e in range(N_iters):
        if ((e+1) % max(1,int(N_iters/10)) == 0):
            print(e, " ", end="", flush=True)

        scores[e] = score_ll(hmm, X)
        hmm = unsupervised_step(hmm, X)
    scores[-1] = score_ll(hmm, X)

    print("\nelapsed", time.time() - start)
    return hmm, scores