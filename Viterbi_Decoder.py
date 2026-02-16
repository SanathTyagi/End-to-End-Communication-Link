import numpy as np

# TRELLIS STRUCTURE DEFINITION
# Defines the valid transitions for a Rate 1/2, K=3 Convolutional Code.
# Format: "Current_State": [ (Parent_State_1, Output_Bits), (Parent_State_2, Output_Bits) ]
# This Look-Up Table (LUT) represents the entire Trellis Diagram.
Transition_Map={
    "00":[("00","00"),("01","11")],
    "10":[("00","11"),("01","00")],
    "01":[("10","01"),("11","10")],
    "11":[("10","10"),("11","11")],
}

State_Index={
    #State:State_index
    "00":0,
    "10":1,
    "01":2,
    "11":3,
}

STATES=["00","10","01","11"]


def get_hamming_distance(received_bits,expected_bits):
    """
    Calculates the Branch Metric (BM).
    Hamming distance = Number of differing bits between received and expected.
    """
    distance=0
    for i in range(len(received_bits)):
        if(received_bits[i]!=expected_bits[i]):
            distance+=1
    return distance


def Minimum_Distance(present_state,prev_cost_array,received_bits):
    """
    Implements the Add-Compare-Select (ACS) Unit.
    
    For a given state, it:
    1. ADD: Adds the Branch Metric to the Previous Path Metric for both parents.
    2. COMPARE: Compares the total costs.
    3. SELECT: Chooses the path with the minimum cost (Survivor Path).
    """
    data=Transition_Map[present_state]
    cost=[]

    # Calculate cost for both possible parent paths
    for i in range(len(data)):
        parent_state=data[i][0]
        prev_cost=0

        # Fetch previous path metric (Accumulated cost up to parent)
        parent_index=State_Index[parent_state]
        prev_cost=prev_cost_array[parent_index]

        # Calculate Branch Metric
        expected_bits=data[i][1]
        # Total Path Metric = Previous Cost + Branch Metric
        cost.append(prev_cost+get_hamming_distance(received_bits,expected_bits))
        
    # Decision: Select the survivor path (Minimum Cost)
    if(cost[0]<=cost[1]):
        # Return (Minimum Cost, Parent Index 0)
        return (cost[0],0)
    else:
        # Return (Minimum Cost, Parent Index 1)
        return (cost[1],1)


def Viterbi_Decoder(estimated_bits):
    """
    Decodes the received bitstream using the Viterbi Algorithm (Maximum Likelihood Estimation).
    """
    # Prepare Input
    received_bits="".join(map(str,estimated_bits))
    N=len(received_bits)

    # Trellis Dimensions
    total_rows=4 # 4 States for K=3

    # Time steps = Total Received Bits * Rate + 1 extra for initial state (t=0)
    # The +1 accounts for the starting column [0, inf, inf, inf]
    total_time=(N//2)+1 
    total_columns=total_time

    # 1. Initialization (Path Metric Matrix)
    # Initialize all costs to Infinity, except the starting state (00) which is 0.
    cost_matrix=np.full((total_rows,total_columns),float('inf'))
    cost_matrix[0][0]=0

    # Survivor Matrix tracks which parent was chosen at each step
    survivor_matrix=np.full((total_rows,total_columns),-1,dtype=int)

    # 2. Forward Pass (Trellis Traversal)
    j=1
    while(j<total_time):
        for i in range(4):
            starting_index=(j-1)*2
            ending_index=starting_index+2
            current_state_name=STATES[i]

            # Compute ACS for this state
            cost_matrix[i][j],survivor_matrix[i][j]=Minimum_Distance(current_state_name,cost_matrix[:,j-1],received_bits[starting_index:ending_index])
        j+=1

    
    # 3. Traceback (Backtracking)
    state_transition_list=[]
    j=total_time-1

    # ASSUMPTION: Zero-Tail Termination (Final state is "00")
    present_state_index=0
    present_state="00"
    state_transition_list.append(present_state)

    while(j>0):
        present_state_path=survivor_matrix[present_state_index][j]
        data=Transition_Map[present_state]
        prev_state=data[present_state_path][0]


        present_state=prev_state
        present_state_index=State_Index[present_state]
        state_transition_list.append(present_state)
        j-=1

    state_transition_list.reverse()


    # 4. Bit Extraction
    # Recover input bit from the state transition (First char of state string)

    a=1
    decoded_bits=[]
    while(a<len(state_transition_list)):
        if(state_transition_list[a][0]=='0'):
            decoded_bits.append(0)
        else:
           decoded_bits.append(1) 
        a+=1
    return decoded_bits
    
    