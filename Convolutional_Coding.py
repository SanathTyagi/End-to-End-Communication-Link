
def Convolutional_Coding(encoded_bits):
    """
    Implements a Rate 1/2, Constraint Length K=3 Convolutional Encoder.
    
    Architecture:
        - Rate (R): 1/2 (2 output bits for every 1 input bit).
        - Constraint Length (K): 3 (Output depends on current bit + 2 previous bits).
    Args:
        encoded_bits (list): The source-coded binary input stream.
        
    Returns:
        convolutional_code (list): The channel-coded bitstream with redundancy added.
    """
    K=3 # Constraint Length

    # 1. Zero Padding (Trellis Termination)
    # Append (K-1) zeros to the input to flush the shift register states back to 0.
    # This ensures the decoder (Viterbi) knows the final state is [0,0].
    padded_input=encoded_bits+[0]*(K-1)
    
    # 2. Initialize Shift Register (Memory)
    # state_memory represents the delay line: [D1, D2]
    # Initially, the register is cleared (all zeros).
    state_memory=[0,0]
    
    convolutional_code=[]

    # 3. Encoding Process
    for bit in padded_input:
        
        # Output 1 Logic: Current Bit XOR 2nd Delay Element
        # Implements connection: Input <-> D2
        convolutional_code.append(bit^state_memory[1])

        # Output 2 Logic: Current Bit XOR 1st Delay XOR 2nd Delay
        # Implements connection: Input <-> D1 <-> D2
        convolutional_code.append(bit^state_memory[0]^state_memory[1])

        # 3. Shift Register Update
        state_memory[1]=state_memory[0]
        state_memory[0]=bit
    return convolutional_code