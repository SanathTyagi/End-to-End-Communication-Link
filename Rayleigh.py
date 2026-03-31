import numpy as np

"""
    Simulates an Rayleigh Fading Channel.
    
    Args:
        transmitted_signal (array): The modulated continuous-time signal.
        SNR_dB (float): Signal-to-Noise Ratio in dB (Eb/N0).
        Eb (float): Energy per bit (used to scale noise relative to signal).
        dt (float): Sampling interval (used for noise bandwidth scaling).
        
    Returns:
        received_signal (array): The received signal with added noise.
        h:rayleigh coeff
    """
def Rayleigh_Channel(transmitted_signal, L_pulse, M, SNR_dB, Eb, dt):
    N=len(transmitted_signal)
    number_of_bits=int(N/L_pulse)
    number_of_symbols=int(number_of_bits/np.log2(M))
    
    SNR_linear=10**(SNR_dB/10)
    No=Eb/SNR_linear
    sigma=np.sqrt(No/(2*dt))
    noise=np.random.normal(0,sigma,size=N)+(1j*(np.random.normal(0,sigma,size=N)))
    
    rayleigh_coeff_amplitude=np.random.rayleigh(scale=1/np.sqrt(2),size=number_of_symbols)
    rayleigh_coeff_phase=np.random.uniform(low=-np.pi,high=np.pi,size=number_of_symbols)
    rayleigh_coeff=rayleigh_coeff_amplitude*np.exp(rayleigh_coeff_phase*1j)
    
    rayleigh_coeff_waveform=np.repeat(rayleigh_coeff, L_pulse)
    received_signal=(rayleigh_coeff_waveform*transmitted_signal)+noise

    return received_signal, rayleigh_coeff_waveform