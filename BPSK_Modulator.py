import numpy as np




def BPSK_modulator(lst,dt,T,A,p):
    """
    Modulates the input bitstream using Binary Phase Shift Keying (BPSK).
    
    Args:
        lst (list): Input bitstream (0s and 1s).
        dt (float): Sampling interval (time step).
        T (float): Symbol duration.
        A (float): Amplitude of the carrier/signal.
        p (array): Pulse shape vector (e.g., Rectangular or SRRC).
        
    Returns:
        t (array): Time vector corresponding to the signal.
        transmitted_test (array): The modulated baseband signal
        M (int): Modulation order (2 for BPSK).
    """

    
    M=2  # Modulation Order (2 points in constellation)
    encoded_bits =np.array(lst)

    
    # 1. Symbol Mapping (Constellation Mapping)
    # Mapping Rule: 
    #   Bit 0 -> +A
    #   Bit 1 -> -A
    # Formula: A * (1 - 2*bit)
    transmitted_bits=A*(1-(2*encoded_bits))

    # 2. Time Vector
    # Total samples = Number of symbols * Samples per symbol
    N=len(transmitted_bits)
    t=np.arange(0,N*T,dt)

    
    # 3. Upsampling (Impulse Train Generation)
    # We place the symbol amplitudes at intervals defined by the pulse length.
    L_pulse=len(p)
    Total=L_pulse*N
    
    impulse_train=np.zeros(Total)
    # Vectorized placement of symbols (skipping L_pulse samples)
    impulse_train[::L_pulse]=transmitted_bits

    # 4. Pulse Shaping
    # Convolve the impulse train with the pulse shape 'p' to generate the continuous waveform.
    # Mode 'full' gives the complete convolution result.
    transmitted_signal=np.convolve(impulse_train,p,mode="full")

    # 5. Truncation
    # We truncate the valid portion to match the time vector length.
    # This aligns the signal strictly within the transmission window [0, N*T].
    transmitted_test=transmitted_signal[:Total]
    return t,transmitted_test,M





