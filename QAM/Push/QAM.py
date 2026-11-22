import numpy as np


#
def PAM_table(m):   
    '''Args: m: number of bits in PAM.
    Returns: table containing key='00', value=-3.
    Purpose: makes a dict of all possible bit pairs in PAM, which are further used 
    in QAM mapping.
    '''
    L = 2**m
    table = {}
    for n in range(L):
        g = n ^ (n >> 1)     # n>>1 means right shift by one bit
        table[format(g, f'0{m}b')] =2*n-(L - 1) # key='00', value=-3
    return table




# Contains key '0000' with value=-A-jA for M-ary QAM.
# Used in the Nearest Neighbour rule during demodulation of QAM.
def QAM_table(m,A):
    '''Args: m: number of bits in PAM (number of bits in QAM divided by 2). 
    In this repo I am considering only even QAM (the number of bits per symbol is even).
    A: amplitude level decided by Es (average energy per symbol) and M.
    Returns: dict with key='0000', value=-A-jA.
    Purpose: used in demodulation while evaluating |r-value| for each constellation point.
    '''
    I=PAM_table(m)
    QAM_dict = {}
    for i_key, i_val in I.items():
        for q_key, q_val in I.items():
            new_key=i_key+q_key
            new_value= A*(i_val + 1j*q_val)
            QAM_dict[new_key]=new_value
    return QAM_dict


def QAM_map(bits_array_2d, M, A):
    '''Args:
    A: amplitude level decided by Es (average energy per symbol) and M.
    M: modulation order (only squares: 4, 16, 64, ...); consider even QAM only. 
    bits_array_2d: a 2-D array such that the number of columns = k, 
    basically each row contains one symbol.
    Returns: array of complex values I+jQ, with a count equal to the total number of rows
    in bits_array_2d.
    Working:
    Takes bits_array_2d and picks each row, then divides it into two sections, which are actually sqrt(M)-PAM
    bit groups. With the help of the PAM table, it maps them using Gray code, builds the whole array for each row,
    and lastly scales it by A.
    '''
    
    k = int(np.log2(M))                   # bits per symbol
    m = k // 2                            # bits per symbol in sqrt(M)-PAM
    table = PAM_table(m)                  # build once, outside the loop

    symbols = []
    for row in bits_array_2d:          # pick one row (one symbol), then divide it into two parts
        i_bits = row[:m]                  # first half  -> I
        q_bits = row[m:]                  # second half -> Q

        i_key = ''.join(map(str, i_bits)) # [0,0]='00'
        q_key = ''.join(map(str, q_bits))

        I = table[i_key]                  # fetch amplitude
        Q = table[q_key]

        symbols.append(I + 1j*Q)          # combine
    return (A*np.array(symbols))

def Nearest_Neighbour(qam_dict,r):
    '''Args:
    qam_dict: dict containing the mapping of QAM keys to constellation points.
    r: received complex value I+jQ after the channel and demodulation.
    Working:
    Finds the distance |r-val| to each constellation point in the QAM,
    stores the distances in distances, takes the argmin of the distances, and returns the key at that position.
    Returns: the estimated key, like '0000'.
    '''
    # list containing the keys of qam_dict
    keys=list(qam_dict.keys())
    # store |r - dict value| in the distances list
    distances = [abs(r - val) for val in qam_dict.values()]
    # take the argmin of distances and report the key at that index.
    return keys[np.argmin(distances)]