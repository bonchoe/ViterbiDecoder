import numpy as np
import pprint

INF = float('inf')

# Define Node class for Viterbi decoder
class Node:
    def __init__(self, state, prob, p = None):
        # state: same as index of instance in list
        self._state   = state
        self._prob    = prob
        self._pointer = p
        
    def set_prob(self, prob):
        self._prob    = prob
        
    def set_pointer(self, pointer):
        self._pointer = pointer
    
    def get_state(self):
        return self._state
    
    def get_prob(self):
        return self._prob
    
    def get_pointer(self):
        return self._pointer
    
    def __repr__(self):
        return "[state: " + str(self._state) + ", prob: " + str(self._prob) + ", pointer: " + str(self._pointer) + "]"

# Viterbi decoder
def viterbi(obs, states, start_p, trans_p, emit_p, debug = False):
    """_summary_

    Args:
        obs (_type_): _description_
        states (_type_): _description_
        start_p (_type_): _description_
        trans_p (_type_): _description_
        emit_p (_type_): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    # List of state metric and backpointer for states
    V = [{}]
    # state_l = len(states)
    

    # Assign state metric and backpointer for every initial state
    # V[0] = [[INF,None]] * state_l
    for st in states:
        try:
            V[0][st] = Node(st, start_p[st] * emit_p[st][obs[0]], None)
        except:
            V[0][st] = Node(st, 0.0, None)
        # if obs[0] not in emit_p[st]:
        #     V[0][st] = Node(st, 0.0, None)
        # else:
        #     V[0][st] = Node(st, start_p[st] * emit_p[st][obs[0]], None)
        
    # pprint.pprint(V[0])
        
    for t in range(1,len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = 0.0
            max_prev    = ""
            for prev_st in states:
                try:
                    tr_prob = V[t-1][prev_st].get_prob() * trans_p[prev_st][st] * emit_p[st][obs[t]]
                    if tr_prob > max_tr_prob:
                        if tr_prob > max_tr_prob:
                            max_tr_prob = tr_prob
                            max_prev    = prev_st
                except:
                    continue
            V[t][st] = Node(st, max_tr_prob, max_prev)
    
    for line in dptable(V):
        print(line)
    
    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, node in V[-1].items():
        if node.get_prob() > max_prob:
            max_prob = node.get_prob()
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 1, 0, -1):
        try:
            previous = V[t][previous].get_pointer()
            opt.insert(0, previous)
        except:
            continue
    # print ("The steps of states are \"" + "\"-\"".join(opt) + "\" with highest probability of %s" % max_prob)
    
    return opt

def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for st in V[0]:
        yield "%.7s: " % st + " ".join("%e" % v[st].get_prob() for v in V)
