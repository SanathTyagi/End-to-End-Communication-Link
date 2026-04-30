import numpy as np

def Demodulator_BPSK(received_signal,p,dt):
    """
    Implements a Coherent BPSK Demodulator.
    
    Architecture:
        - Instead of full convolution (Matched Filter), we slice the signal into symbol periods.
        - We perform a dot product with the pulse shape to integrate the energy.
        - This assumes perfect timing synchronization (Ideal Receiver).
        
    Args:
        received_signal (array): The noisy baseband signal.
        p (array): The pulse shape used at the transmitter.
        dt (float): Sampling interval.
        
    Returns:
        output_sampled (array): Soft decision variables (energy estimates).
        estimated_bits (array): Hard decision bits (0 or 1).
    """

    
    L_pulse=len(p)

    # 1. Serial-to-Parallel Conversion (Reshaping)
    # We break the long 1D signal array into a 2D matrix where:
    #   Rows = Number of Symbols
    #   Columns = Samples per Symbol (L_pulse)
    # This prepares the data for vectorized integration.
    received_signal_matrix=np.array(received_signal).reshape(-1,L_pulse)


    # 2. Matched Filter Operation
    # Instead of convolving, we compute the dot product of each row with the pulse 'p'.
    # Multiplying by 'dt' completes the integration: Integral(r(t) * p(t) dt)
    # This gives the value of the Matched Filter at the sampling instant t=T.
    output_sampled=np.dot(received_signal_matrix,p)*dt

    # 3. Decision Rule (Hard Detection)
    # The decision boundary for BPSK is 0.
    #   If Energy > 0 --> Estimate Bit 0 (Mapped to +A)
    #   If Energy < 0 --> Estimate Bit 1 (Mapped to -A)
    estimated_bits=np.where(np.real(output_sampled)>=0,0,1)

    return output_sampled,estimated_bits
