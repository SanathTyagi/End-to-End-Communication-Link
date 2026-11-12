import numpy as np

"""
    Simulates an Equalizer.
    Type: Minimum Mean Square Error (MMSE)
    Why MMSE why not ZF?
    At low SNR, probability of deep fade increases due to this, in ZF(Zero Forcing) eqaulizer at deep fade, noise amplifies violently.
    In MMSE at high SNR, it works same as ZF and at low SNR, division by No restrict the amplfication of noise.
    Args:
        received_signal (array): The received signal with added noise.(y=xh+n)
        rayleigh coeff: h (Currently i am considering ideal Channel Estimation)
    Returns:
        equalised received_signal: y=y*w.
    """
def Equalizer(received_signal, h, SNR_dB, Eb):
    SNR_linear=10**(SNR_dB/10)
    No=Eb/SNR_linear
    w=np.conj(h)/((np.abs(h)**2)+No)
    equalised_received_signal=received_signal*w
    return equalised_received_signal