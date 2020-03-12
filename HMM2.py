"""HMM implementation, adapated from assignment 6"""

import numpy as np
from collections import namedtuple
import time

Hmm = namedtuple("Hmm", ["A0", "A", "O", "D", "L"])

def init(A, O):
    A, O = np.array(A), np.array(O)
    L, D = O.shape
    return Hmm(np.ones(L)/L, A, O, D, L)

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
    seq = ""
    state = np.argmax(probs[-1])
    for i in range(M):
        seq = str(int(state)) + seq
        state = steps[-i-1,int(state)]

    return seq



def forward(hmm, x, normalize=False):
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
    alphas[0] = np.array(hmm.A0) * O[:,x[0]]
    if (normalize):
        alphas[0] /= np.sum(alphas[0])

    # for every element of x
    for i in range(1,M):
        # for every possible next state
        for z in range(hmm.L):
            alphas[i,z] = O[z,x[i]] * np.sum(alphas[i-1] * A[:,z])
        if (normalize):
            alphas[i] /= np.sum(alphas[i])

    return alphas


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
    datset (X, Y). Note that this method does not return anything, but
    instead updates the attributes of the HMM object.

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


def unsupervised_learning(hmm, X, N_iters):
    '''
    Trains the HMM using the Baum-Welch algorithm on an unlabeled
    datset X. Note that this method does not return anything, but
    instead updates the attributes of the HMM object.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of length M, consisting of integers ranging
                    from 0 to D - 1. In other words, a list of lists.

        N_iters:    The number of iterations to train on.
    '''
    A = np.array(hmm.A)
    O = np.array(hmm.O)

    print("Training", N_iters, "iters")
    start = time.time()
    # epochs
    for e in range(N_iters):
        if ((e+1) % (int(N_iters/10)) == 0):
            print(e, " ", end="", flush=True)

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
                Pab = np.outer(alphas[i], betas[i+1]) * A * O[:,x[i+1]]
                A_num += Pab / np.sum(Pab)
                # probability of observation
                O_num[:,x[i]] += Pa[i]
            O_num[:,x[-1]] += Pa[-1]

        # update
        A = A_num / A_den.reshape((-1,1))
        O = O_num / O_den.reshape((-1,1))
        hmm = hmm._replace(A=A, O=O)

    print("\nelapsed", time.time() - start)
    return hmm


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


def probability_alphas(hmm, x):
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
    alphas = forward(hmm, x)

    # alpha_j(M) gives the probability that the state sequence ends
    # in j. Summing this value over all possible states j gives the
    # total probability of x paired with any state sequence, i.e.
    # the probability of x.
    prob = sum(alphas[-1])
    return prob


import random
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

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = init(A, O)
    return supervised_learning(HMM, X, Y)

def unsupervised(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = init(A, O)
    return unsupervised_learning(HMM, X, N_iters)
