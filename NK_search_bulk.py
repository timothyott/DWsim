print(''' ----------------------------------------------------
NK search (Version 1.0)

----------------------------------------------------

By Ron Tidhar & Tim Ott

----------------------------------------------------''')

'''
Given a set of NK landscapes (see NK_Model_basic.py), 
this script applies local search, decision weaving search (Ott & Eisenhardt,2016), and chunky search (Baumann & Siggelkow, 2013) strategies
Outputting the results with errors presented as one standard deviation
'''

# *** IMPORTS ***

import numpy as np
import itertools
from time import time
import os
import sys
import scipy.stats as st
import random

def main(inputFile, runs, thresh, chunky, ss_prob):    
    try:
        NK = np.load(inputFile)
    except IndexError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)

    '''
    NK has dimensions (i, 2**N, N*2+2), where
      i     = number of landscapes
      2**N  = the number of possible decision sets
      N*2+2 = the value of each point on each landscape.
                - the first N columns are for the combinations of N decision variables (DV)
                    (hence we have 2**N rows for the total number of possible combinations)
                - the second N columns are for the contribution values of each DV
                - the next valuer is for the total fit (avg of N contributions)
                - the last one is to find out whether it is the local peak (0 or 1)
    '''
    chunk_flags = {1:"chunky", 2:"incremental", 3:"integrated"}

    runs = runs
    threshold = thresh
    chunk_key = chunky
    ss_prob = ss_prob
    
    filename = str(inputFile)
    print("Simulating "+filename+"...", end=" ")
    P = int(filename[filename.find("P_")+len("P_")])
    D = int(filename[filename.find("D_")+len("D_")])
    Kw = float(filename[filename.find("within_")+len("within_"):filename.find("_K_between")])
    Kb = float(filename[filename.find("between_")+len("between_"):filename.find("_i_")])
    Domain_decision_set = domain_dec_set(D)
    i = NK.shape[0]
    N = int(np.log2(NK.shape[1]))
    Power_key = powerkey(N)

    num_local_steps = []
    num_local_scans = []
    final_local_fitness = []
    norm_local_fitness = []
    final_local_diff = []
    nonglobal_local_fit = []
    local_global_count = 0
    
    num_dw_steps = []
    num_dw_steps_per_domain = []
    num_dw_scans = []
    final_dw_fitness = []
    norm_dw_fitness = []
    final_dw_diff = []
    nonglobal_dw_fit = []
    dw_global_count = 0

    num_chunky_steps = []
    num_chunky_scans = []
    final_chunky_fitness = []
    norm_chunky_fitness = []
    final_chunky_diff = []
    nonglobal_chunky_fit = []
    chunky_global_count = 0

    for land in range(i): 
        max_fit = max(NK[land,:,2*N])
        
        for _ in range(runs):
            start = random_start(N)
            start_fit = fitness(start, N, NK[land], Power_key)
            norm_start_fit = start_fit/max_fit

            (local_steps, local_scans, local_fit) = local_search(start, N, NK[land], Power_key)
            (dw_steps, dw_scans, dw_fit, dw_steps_per_domain) = decision_weaving(start, N, P, D, NK[land], Power_key, Domain_decision_set, threshold, ss_prob)
            (chunky_steps, chunky_scans, chunky_fit) = chunky_search(start, N, NK[land], Power_key, chunk_flags[chunk_key])


            num_local_steps.append(local_steps)
            num_local_scans.append(local_scans)

            final_local_fitness.append(local_fit)

            norm_local_fit = local_fit / max_fit
            norm_local_fitness.append(norm_local_fit)
            final_local_diff.append(norm_local_fit - norm_start_fit)
            
            if local_fit == max_fit:
                local_global_count += 1
            else:
                nonglobal_local_fit.append(norm_local_fit)


            num_dw_steps.append(dw_steps)
            num_dw_steps_per_domain.append(dw_steps_per_domain)
            num_dw_scans.append(dw_scans)

            final_dw_fitness.append(dw_fit)

            norm_dw_fit = dw_fit/max_fit
            norm_dw_fitness.append(norm_dw_fit)
            final_dw_diff.append(norm_dw_fit - norm_start_fit)

            if dw_fit == max_fit:
                dw_global_count += 1
            else:
                nonglobal_dw_fit.append(norm_dw_fit)

            num_chunky_steps.append(chunky_steps)
            num_chunky_scans.append(chunky_scans)
            
            final_chunky_fitness.append(chunky_fit)
            
            norm_chunky_fit = chunky_fit/max_fit
            norm_chunky_fitness.append(norm_chunky_fit)
            final_chunky_diff.append(norm_chunky_fit - norm_start_fit)
            
            # to recreate the table that B&S have need other variables
            if chunky_fit == max_fit:
                chunky_global_count += 1
            else:
                nonglobal_chunky_fit.append(norm_chunky_fit)

    print("Finished!")
    
    #Really what we want to be doing is returning the info so that it can be output later.
    result_dict =   { 
                    'P': P, 
                    'D': D, 
                    'Kw': Kw, 
                    'Kb': Kb, 
                    'threshold': threshold, 
                    'ss_prob': ss_prob
                    }
    
    local_dict = compile_search_results("local", norm_local_fitness, final_local_diff, local_global_count, i, runs, nonglobal_local_fit, num_local_steps, num_local_scans)
    dw_dict = compile_search_results("dw", norm_dw_fitness, final_dw_diff, dw_global_count, i, runs, nonglobal_dw_fit, num_dw_steps, num_dw_scans)
    chunky_dict = compile_search_results("chunky", norm_chunky_fitness, final_chunky_diff, chunky_global_count, i, runs, nonglobal_chunky_fit, num_chunky_steps, num_chunky_scans)
    
    steps_per_domain_array = np.array(num_dw_steps_per_domain)

    dw_dict['dw_steps_per_domain'] = np.mean(steps_per_domain_array, axis=0)
    dw_dict['dw_steps_per_domain_conf_int'] = np.apply_along_axis(conf_int_hw, 0, steps_per_domain_array)

    full_results_dict = merge_dicts(result_dict, local_dict, dw_dict, chunky_dict)

    return(full_results_dict)
 

# FUNCTIONS

def powerkey(N):
    '''
    Used to find the location on the landscape for a given combination
    of the decision variables. Maps a decision string to the correct row in the NK_land matrix.
    Returns a vector with the powers of two: (2^(N-1), ..., 1)
    '''
    Power_key = np.power(2, np.arange(N - 1, -1, -1))
    return(Power_key)

def calc_fit(N, NK_land, inter_m, Current_position, Power_key):
    '''
    Takes landscape and a combination and returns a vector of fitness
    values (contribution value for each of the N decision variables)
    '''
    Fit_vector = np.zeros(N)
    for ad1 in np.arange(N):
        Fit_vector[ad1] = NK_land[np.sum(Current_position * inter_m[ad1] * Power_key), ad1]
    return(Fit_vector)

def random_start(N):
    return np.random.randint(2, size=N)

def local_search(dec, N, NK, Power_key):
    new_dec = dec.copy()
    new_dec[0] = abs(dec[0] - 1)
    stepped = 0
    scanned = 1 #you scan your random spot before you start moving

    while(True):
        stepped += 1
        (new_dec, scans) = local_step(N, NK, dec, Power_key)
        scanned += scans
        if (all(new_dec == dec)):
            dec = new_dec
            break
        else:
            dec = new_dec
    return stepped, scanned, fitness(dec, N, NK, Power_key)

def local_step(N, NK, Current_position, Power_key):
    ''' 
    Local search strategy operates by changing only one decision 
    to reach a better position in the solution space.
    Thus, for a given NK instance and decision set, 
    we randomly iterate through neighbours, and go to the first, better option
    (In keeping with Levinthal, 1997, we don't assume the agent goes to the 
    highest-valued neighbour)
    '''
    scans = 0

    # first make sure we're not at a local peak (if yes, we're done)
    if not local_max(Current_position, N, NK, Power_key):
        Indexes = np.arange(N)
        np.random.shuffle(Indexes)

        Current_fit = fitness(Current_position, N, NK, Power_key)
        New_position = Current_position.copy()

        for new_dec in Indexes:  # check for the neighbourhood
            scans += 1
            New_position[new_dec] = abs(New_position[new_dec] - 1)
            
            if (fitness(New_position, N, NK, Power_key) > Current_fit):
                # We have found a better position          
                return (New_position, scans)
            # We didn't find a better position => change decision back
            New_position[new_dec] = abs(New_position[new_dec] - 1)

    # If we're here, we must be at a local optimum
    return (Current_position, scans)

def decision_weaving(Current_position, N, P, D, NK, Power_key, Domain_decision_set, threshold, ss_prob):
    unvisited_policies = np.arange(P)
    stepped = 0
    scanned = 1 #you scan the position you start in
    steps_per_domain = []

    np.random.shuffle(unvisited_policies)
    for policy in unvisited_policies:
        (New_position, steps, scans) = search_domain(N, P, D, NK, Current_position, policy, Power_key, Domain_decision_set, threshold, ss_prob)
        stepped += steps
        scanned += scans
        steps_per_domain.append(steps)
        
        Current_position = New_position.copy()

    return stepped, scanned, fitness(Current_position, N, NK, Power_key), steps_per_domain

def search_domain(N, P, D, NK, Current_position, policy, Power_key, Domain_decision_set, learning_threshold, ss_prob):
    New_position = Current_position.copy()
    random.shuffle(Domain_decision_set)
    focus_steps = 0
    focus_scans = 0
    ss_steps = 0

    for pp in Domain_decision_set:
    # check for other decision sets within domain (policy can change)
        if not (all(pp == Current_position[policy*D:(policy+1)*D])):
            focus_scans += 1
            New_position = Current_position.copy()
            New_position[policy*D:(policy+1)*D] = pp

            if (fitness(New_position, N, NK, Power_key) > fitness(Current_position, N, NK, Power_key)):
                # We have found a better position      
                focus_steps += 1
                Current_position = New_position.copy()

            #Randomly look for a stepping stone, not always
            if (np.random.binomial(1,ss_prob)):
                #Stepping stone block
                New_position = stepping_stone(N, P, D, NK, Current_position, policy, Power_key)
                if not (all(Current_position == New_position)):
                    ss_steps += 1
                    Current_position = New_position.copy()
            
            # Want to switch domains if we have made more improvements in this domain than the threshold
            if (focus_steps >= learning_threshold):
                break

    return Current_position, focus_steps+ss_steps, focus_scans

def stepping_stone(N, P, D, NK, Current_position, policy, Power_key):
    #This is the first iteration of stepping stones, where we randomly look at a new position in any domain
    Current_fit = fitness(Current_position, N, NK, Power_key)
    New_position = Current_position.copy()

    stepping_stones = np.delete(range(P*D), list(range(policy*D, (policy+1)*D)))
    np.random.shuffle(stepping_stones)
    dec = stepping_stones[0]

    New_position[dec] = abs(New_position[dec] - 1)
            
    if (fitness(New_position, N, NK, Power_key) > Current_fit):
        # We have found a better position          
        return New_position
    else:
        return Current_position
    
def stepping_stone2(N, P, D, NK, Current_position, policy, Power_key):
    #This is the first iteration of stepping stones, where we choose the best overall performance of the 4 possible choice sets in a given domain
    Current_fit = fitness(Current_position, N, NK, Power_key)
    temp_position = New_position = Current_position.copy()
    max_fit =0
    
    #Create a list of the domains that are not currently in focus
    ss_domains = np.delete(np.arange(P),policy)
    #randomly pick a domain
    np.random.shuffle(ss_domains)
    ss_domain = ss_domains[0]
    #loop through chosen domain and look for best possible 1 decision change that can be made
    for dec in list(range(ss_domain*D, (ss_domain+1)*D)):
        temp_position[dec] = abs(New_position[dec] - 1)
        if (fitness(temp_position, N, NK, Power_key) > max_fit):
            max_fit = fitness(temp_position, N, NK, Power_key)
            ss_dec = dec
    # after finding max of the possible SS, set new position to that and return
    New_position[ss_dec] = abs(New_position[ss_dec] - 1)
    #return New_position
    if (fitness(New_position, N, NK, Power_key) > Current_fit):
        # We have found a better position          
        return New_position
    else:
        return Current_position
    
def chunky_search(Current_position, N, NK, Power_key, flag="chunky"):
    # 1. set chunk order: chunky = {5+1+...+1}, incremental = {1+...+1}, integrated = {9}
    # 2. local search first chunk
    # 3. converge
    # 4. expand chunk
    # 5. repeat until chunk = N (i.e., integrated - this is just local search)

    if (flag == "chunky"):
        chunks = [int((N+1)/2)] + [1 for i in range(N - int((N+1)/2))]
    elif (flag == "incremental"):
        chunks = [1 for i in range(N)]
    elif (flag == "integrated"):
        # chunks = [N] (this local search)
        return local_search(Current_position, N, NK, Power_key)
    else:
        print("ERROR: Please provide a correct flag to indicate chunky search type")

    stepped = 0
    scanned = 1 # you evaluate your initial starting position

    for j in range(len(chunks)):
        New_position = Current_position.copy()
        New_position[0] = abs(Current_position[0] - 1)
        chunk_ind = np.sum(chunks[:j+1])

        while(True):
            stepped += 1
            (New_position, scans) = chunk_local_step(N, NK, Current_position, chunk_ind, Power_key)
            scanned += scans
            if (all(New_position == Current_position)):
                Current_position = New_position
                break
            else:
                Current_position = New_position

    # At this point, we have an integrated search domain, which is simply local search
    (integrated_step, integrated_scan, fitness) = local_search(Current_position, N, NK, Power_key)
    stepped += integrated_step
    scanned += integrated_scan

    return stepped, scanned, fitness

def chunk_local_step(N, NK, Current_position, chunk_ind, Power_key):
    scans = 0

    Indexes = np.arange(chunk_ind)
    np.random.shuffle(Indexes)

    Current_fit = chunk_fitness(Current_position, chunk_ind, N, NK, Power_key)
    New_position = Current_position.copy()

    for new_dec in Indexes:  # check for the neighbourhood
        scans += 1
        New_position[new_dec] = abs(New_position[new_dec] - 1)
        
        if (chunk_fitness(New_position, chunk_ind, N, NK, Power_key) > Current_fit):
            # We have found a better position          
            return (New_position, scans)
        # We didn't find a better position => change decision back
        New_position[new_dec] = abs(New_position[new_dec] - 1)

    # If we're here, we must be at a local (parochial) optimum
    return (Current_position, scans)

def current_policy(P, D, decision):
    policy = []
    for pp in range(P):
        pol = 0
        for dec in range(D):
            pol += decision[pp*D + dec]
        
        policy.append(int(int(pol) > int(D/2))) # Policy is defined by majority decisions

    return policy

def fitness(position, N, NK, Power_key):
    return NK[np.sum(position*Power_key), 2*N]

def local_max(position, N, NK, Power_key):
    return NK[np.sum(position*Power_key), 2*N+1]

# def max_fitness(NK, N, Power_key):
#     #Need to change this to utilize local_max column in landscape matrix
#     # max_fit=0
#     # for dec in itertools.product(range(2), repeat=N):
#     #     curr_fit = fitness(dec, N, NK, Power_key)
#     #     if curr_fit > max_fit:
#     #         max_fit = curr_fit

#     return max(NK[:,2*N])

def chunk_fitness(position, num_dec_in_chunk, N, NK, Power_key):
    return np.mean(NK[np.sum(position*Power_key), N:(N+num_dec_in_chunk)])

def print_num_steps(steps, indent = "     "):
    print_str = indent + "Average # of steps  = "
    print_str += conf_interval(steps)

    print(print_str)

def print_num_domain_steps(steps, indent = "     "):
    l =np.asarray(steps).T.tolist()
    for index, item in enumerate(l):
        print_str = indent + "Average # of steps in domain " + str(index+1) + " = "
        print_str += conf_interval(item)

        print(print_str)
        
def print_num_scans(steps, indent = "     "):
    print_str = indent + "Average # of scans  = "
    print_str += conf_interval(steps)

    print(print_str)

def print_fitness(fitness, norm_fitness, diff, indent = "     "):
    #print_str = indent + "Average fitness     = "
    #print_str += conf_interval(fitness)

    #print(print_str)
    
    print_str = indent + "Normalized fitness     = "
    print_str += conf_interval(norm_fitness)

    print(print_str)
    
    print_str = indent + "Average improvement = "
    print_str += conf_interval(diff)
    
    print(print_str)
    
def print_fitness_peak(percent_global, norm_nonglobal_fitness, indent = "     "):
    print_str = indent + "% teams at global peak     = "
    print_str += str(percent_global*100)

    print(print_str)
    
    print_str = indent + "Nonglobal fitness     = "
    print_str += conf_interval(norm_nonglobal_fitness)

    print(print_str)

def conf_interval(values, delta=0.05):
    z = st.norm.ppf(1-delta/2)
    point_est = np.mean(values)    
    hw = z * np.sqrt(np.var(values, ddof=1) / len(values))
    c_low = point_est - hw
    c_high = point_est + hw
    return "%.4f with %.f%% Confidence Interval [%.4f" %(point_est, (1-delta)*100, c_low) +  ", %.4f" %c_high + "]"

def conf_int_hw(values, delta=0.05):
    z = st.norm.ppf(1-delta/2)
    point_est = np.mean(values)    
    hw = z * np.sqrt(np.var(values, ddof=1) / len(values))
    #c_low = point_est - hw
    #c_high = point_est + hw
    return hw

def domain_dec_set(D):
    dec_set = []
    for dec in itertools.product(range(2), repeat=D):
        dec_set.append(dec)

    random.shuffle(dec_set)
    return dec_set

def compile_search_results(search_strat, norm_fitness, final_perf_diff, global_count, i, runs, nonglobal_fit, num_steps, num_scans):
    results_dict = {
                    search_strat + '_norm_fit': np.mean(norm_fitness), 
                    search_strat + '_fit_conf': conf_int_hw(norm_fitness), 
                    search_strat + '_avg_impr': np.mean(final_perf_diff), 
                    search_strat + '_impr_conf': conf_int_hw(final_perf_diff), 
                    search_strat + '_pct_global': (global_count/(i*runs)), 
                    search_strat + '_nonglobal_fit': np.mean(nonglobal_fit), 
                    search_strat + '_nonglobal_conf': conf_int_hw(nonglobal_fit), 
                    search_strat + '_avg_steps': np.mean(num_steps), 
                    search_strat + '_step_conf': conf_int_hw(num_steps), 
                    search_strat + '_avg_scans': np.mean(num_scans), 
                    search_strat + '_scan_conf': conf_int_hw(num_scans)
                    }
    return results_dict

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
