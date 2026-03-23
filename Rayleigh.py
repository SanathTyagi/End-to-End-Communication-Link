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
def Rayleigh_Channel(transmitted_signal, SNR_dB, Eb, dt):
    
    N=len(transmitted_signal)
    SNR_linear=10**(SNR_dB/10)
    No=Eb/SNR_linear
    sigma=np.sqrt(No/(2*dt))
    noise=np.random.normal(0,sigma,size=N)+(1j*(np.random.normal(0,sigma,size=N)))
    
    rayleigh_coeff_amplitude=np.random.rayleigh(scale=1/np.sqrt(2),size=N)
    rayleigh_coeff_phase=np.random.uniform(low=-np.pi,high=np.pi,size=N)
    rayleigh_coeff=rayleigh_coeff_amplitude*np.exp(rayleigh_coeff_phase*1j)
    
    received_signal=(rayleigh_coeff*transmitted_signal)+noise

    return received_signal, rayleigh_coeff