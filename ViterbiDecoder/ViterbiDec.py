import numpy as np
INF = float('inf')

# Define Node class for Viterbi decoder
class Node:
    def __init__(self, state, sm = INF):
        # state: same as index of instance in list
        self._state   = state
        self._sm      = sm
        self._pointer = None
        
    def set_sm(self, sm):
        self._sm      = sm
        
    def set_pointer(self, pointer):
        self._pointer = pointer
    
    def get_state(self):
        return self._state
    
    def get_sm(self):
        return self._sm
    
    def get_pointer(self):
        return self._pointer
    
    def __repr__(self):
        return "[state: " + str(self._state) + ", sm: " + str(self._sm) + ", pointer: " + str(self._pointer) + "]"

def encode(bits_in, generators):
    """ Encode input in convolution code

    Args:
        bits_in (list): bit sequence to encode in convolution code
        generators (list): list of list, elements of which implies bits of generator

    Returns:
        out (list): encoded bit sequence
    """
    
    l = max(map(len, generators))
    
    # Initialize shift register window for convolution
    bits = [0] * l
    out = []
    
    for i in range(len(bits_in)):
        # Push bits to the right so that [x[n], x[n-1], x[n-2]]
        bits = [bits_in[i]] + bits[:-1]
        
        # Do inner product with generator to choose bit digit to add mod 2
        # and append to the output
        for g in generators:
            out.append(
                np.inner(bits, g) % 2
            )
    return out

# Pass encoded bits through BSC
def bsc(bits_in, p=0.05):
    """ Pass encoded bits through BSC

    Args:
        bits_in (list): Tx bit sequence
        p (float, optional): probability for BSC. Defaults to 0.05.

    Returns:
        out (list): Tx bit sequence with channel noise
    """
    out = []
    for b in bits_in:
        if np.random.uniform() > p:
            out.append(b)
        else:
            out.append(1-b)
    return out

# Viterbi decoder
def viterbi_dec(rx, FSM, last_state = 0b00, debug = False):
    """ 

    Args:
        rx (list): Rx bit sequence from BSC channel
        FSM (list): list of list predefined for convolution code
        last_state (int, optional): _description_. Defaults to 0b00.

    Returns:
        decoded (list), error (int): decoded sequence by Viterbi decoder and its error
    """
    # Adding overhead to flush
    rx = rx
    state_num = len(FSM)
    
    # List of state metric and backpointer for states
    V = [[]]
    
    decoded = []

    # Assign state metric and backpointer for every initial state
    # V[0] = [[INF,None]] * state_num
    V[0] = [Node(s) for s in range(state_num)]

    # Set state metric to ZERO only for last state
    V[0][last_state].set_sm(0)
    
    # i: time index
    # including bits from flushing
    for i in range(1,int(len(rx) / 2) + 1):
        # Append states for every untreated states
        V.append([Node(s) for s in range(state_num)])
        
        # (1) branch metrics unit
        # (2) ACS cell in Viterbi Decoder
        # j, state: states (and its index j) in previous time index (i-1)
        for j, state in enumerate(V[i-1]):
            if state.get_sm() != INF:
                # k, bm: destination state of branch and its branch metric
                for k, code in enumerate(FSM[j]):
                    if code != None:
                        # Calculating branch metric
                        bm = ((code // 2) ^ rx[2*i-2]) + ((code % 2) ^ rx[2*i-1])
                        metric = state.get_sm() + bm
                        if debug == True:
                            print("Updating state metric...")
                            print("i = ", i, ", j = ", j, ", k = ", k, ", metric = ", metric)
                        if V[i][k].get_sm() > metric:
                            # state metric
                            V[i][k].set_sm(metric)
                            # backpointer
                            V[i][k].set_pointer(state)
    # for i in range(len(V)):
    #     print(V[i])
    
    
    min_sm = 0
    min_sm_idx = 0
    last = len(V) - 1
    # Find state index in the last index s.t. sm is minimum
    # But we should implement it such that it finds multiple indices when errors are same
    for s in range(state_num):
        sm = V[last][s].get_sm()
        if debug == True:
            print(s,":", sm)
        if V[last][min_sm_idx].get_sm() > sm:
            min_sm_idx = s
        # elif V[last][min_sm_idx].get_sm() == sm:
    if debug == True:
        print(min_sm_idx)
    
    # Access the state node instance
    last_state = V[last][min_sm_idx]
    
    # state metric of the last state
    # i.e., 
    error = last_state.get_sm()

    # (3) Survivor path decode
    tracker = last_state
    while tracker.get_pointer() != None:
        decoded.append(tracker.get_state() // 2)
        tracker = tracker.get_pointer()

    decoded.reverse()

    return decoded, error
