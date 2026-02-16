import numpy as np


def AWGN_Channel(transmitted_signal,SNR_dB,Eb,dt):
    """
    Simulates an Additive White Gaussian Noise (AWGN) Channel.
    
    Args:
        transmitted_signal (array): The modulated continuous-time signal.
        SNR_dB (float): Signal-to-Noise Ratio in dB (Eb/N0).
        Eb (float): Energy per bit (used to scale noise relative to signal).
        dt (float): Sampling interval (used for noise bandwidth scaling).
        
    Returns:
        received_bits (array): The received signal with added noise.
    """

    N=len(transmitted_signal)
    
    # 1. Convert SNR from dB to Linear scale
    # Formula: SNR_linear = 10^(SNR_dB / 10)
    SNR_linear=10**(SNR_dB/10)

    # 2. Calculate Noise Power Spectral Density (N0)
    # Since SNR = Eb / N0, we have N0 = Eb / SNR_linear
    No=Eb/SNR_linear

    # 3. Calculate Noise Standard Deviation (Sigma)
    # The noise variance depends on the sampling bandwidth (approx 1/dt).
    # Formula: sigma = sqrt(N0 / (2 * dt))
    # This scaling ensures the noise power is correct for the simulation time step.
    sigma=np.sqrt(No/(2*dt))

    # 4. Generate Noise Vector
    # Creates Gaussian noise with Mean = 0 and Std Dev = sigma
    noise=np.random.normal(0,sigma,size=N)

    # 5. Add Noise to Signal
    received_bits=transmitted_signal+noise
    return received_bits


