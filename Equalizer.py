import numpy as np

"""
    Simulates an Equalizer.
    
    Args:
        received_signal (array): The received signal with added noise.(y=xh+n)
        rayleigh coeff: h
    Returns:
        equalised received_signal: y/h=x+n/h
    """
def Equalizer(received_signal, h, SNR_dB, Eb):
    SNR_linear=10**(SNR_dB/10)
    No=Eb/SNR_linear
    w=np.conj(h)/((np.abs(h)**2)+No)
    equalised_received_signal=received_signal*w
    return equalised_received_signal