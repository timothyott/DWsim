print(''' ----------------------------------------------------
NK model (Version 3.2)

----------------------------------------------------

By Ron Tidhar & Tim Ott - adapted from Maciej Workiewicz (2014)

----------------------------------------------------''')

'''
This script creates i (i=100) NK landscapes (with the number of Policies (P) and Decisions per policy (D) and K chosen by the user)
The NK landscapes are saved as a binary file (.npy)
'''

# *** IMPORTS ***

import numpy as np
import itertools
from time import time
import os

def main():    
    # *** MODEL INPUTS ***

    # SYSTEM INPUT
    i = 100  # number of landscapes to produce (use 100 or less for testing)

    # USER INPUTS

    cwd = os.getcwd()
    cwd += "/"
    print("Saving outputs to ", cwd)

    P = int(input("Enter a value for P, the number of policies: "))
    D = int(input("Enter a value for D, the number of decisions within a policy: "))
    
    N = int(P*D)

    K_within = float(input("Input K_within (decimal from 0 to 1): "))
    K_between = float(input("Input K_between (decimal from 0 to 1): "))
    
    assert(0 <= K_within <= 1)
    assert(0 <= K_between <= 1)

    Int_matrix = matrix_rand(N, P, D, K_within, K_between)

    # *** GENERAL VARIABLES AND OBJECTS ***

    # For each iteration i

    Power_key = powerkey(N)
    NK = np.zeros((i, 2**N, N*2+2))
    for i_1 in np.arange(i):
        '''
        First create the landscapes for the DDs and PDs
        '''
        NK_land = nkland(N)
        NK[i_1] = comb_and_values(N, NK_land, Power_key, Int_matrix)    

    filename = 'NK_landscape_P_' + str(P) + '_D_' + str(D) + '_K_within_' + str(K_within) + \
            '_K_between_' + str(K_between) + '_i_' + str(i) + '.npy'
    np.save(cwd + filename, NK)
    print("Saved landscapes as file " + filename)
    '''
    This saves the landscape into a numpy binary file
    '''

# FUNCTIONS AND INTERACTION MATRIX

def matrix_rand(N, P, D, K_within, K_between):
    '''
    This function takes the number of policies, decisions and interdependencies 
    within policies (according to the probability K_within) and 
    across policies (according to the probability K_between). 
    It then creates a random interaction matrix with a diagonal filled.
    '''

    Int_matrix_rand = np.zeros((int(P*D), int(P*D)))
    for i in range(int(P*D)):
        for j in range(int(P*D)):
            if (i == j):
                Int_matrix_rand[i,j] = 1
            elif (int(i/D) == int(j/D)): # within same policy
                Int_matrix_rand[i,j] = int(np.random.rand(1) < K_within)
            else:
                Int_matrix_rand[i,j] = int(np.random.rand(1) < K_between)
    return(Int_matrix_rand)

def powerkey(N):
    '''
    Used to find the location on the landscape for a given combination
    of the decision variables. Maps a decision string to the correct row in the NK_land matrix.
    Returns a vector with the powers of two: (2^(N-1), ..., 1)
    '''
    Power_key = np.power(2, np.arange(N - 1, -1, -1))
    return(Power_key)


def nkland(N):
    '''
    Generates an NK landscape - an array of random numbers ~U(0, 1).
    '''
    NK_land = np.random.rand(2**N, N)
    return(NK_land)

def comb_and_values(N, NK_land, Power_key, inter_m):
    '''
    Calculates values for *all combinations* on the landscape.
    - the first N columns       = combinations of N decision variables (DV)
                                    (hence we have 2**N rows for the total number of possible combinations)
    - the second N columns      = contribution values of each DV
    - the next column           = total fit (avg of N contributions)
    - the last column (bool)    = the point is a local peak (0 or 1)
    '''
    Comb_and_value = np.zeros((2**N, 2*N+2))
    c1 = 0  # starting counter for location

    # iterator creates bit strings for all possible combinations (i.e. positions)
    for c2 in itertools.product(range(2), repeat=N):
        '''
        this takes time so careful
        '''
        Combination1 = np.array(c2)  # taking each combination (convert iterator to an array)
        fit_1 = calc_fit(N, NK_land, inter_m, Combination1, Power_key)
        Comb_and_value[c1, :N] = Combination1  # combination and values
        Comb_and_value[c1, N:2*N] = fit_1
        Comb_and_value[c1, 2*N] = np.mean(fit_1)
        c1 = c1 + 1

    # now let's see if it's a local peak
    for c3 in np.arange(2**N):  
        Comb_and_value[c3, 2*N+1] = 1  # assume it is
        for c4 in np.arange(N):  # check for the neighbourhood
            new_comb = Comb_and_value[c3, :N].copy()
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((Comb_and_value[c3, 2*N] < Comb_and_value[int(np.sum(new_comb*Power_key)), 2*N])):
                Comb_and_value[c3, 2*N+1] = 0  # if smaller than the neighbour then not peak
                break
    return(Comb_and_value)

def calc_fit(N, NK_land, inter_m, Current_position, Power_key):
    '''
    Takes landscape and a combination and returns a vector of fitness
    values (contribution value for each of the N decision variables)
    '''
    Fit_vector = np.zeros(N)

    for ad1 in np.arange(N):
        Fit_vector[ad1] = NK_land[int(np.sum(Current_position * inter_m[ad1] * Power_key)), ad1]
    return(Fit_vector)

def random_start(N):
    return np.random.randint(2, size=N)

if __name__ == '__main__':
    main()
