import numpy as np

"""
    Simulates a slow, flat Rayleigh fading channel: r(t) = h(t)*s(t) + n(t)

    Fading is slow — h is drawn once per symbol and held constant across all
    L_pulse samples within that symbol.

    Each coefficient h = alpha * exp(j*phi), where:
        alpha ~ Rayleigh(1/sqrt(2))   (unit mean power)
        phi   ~ Uniform(-pi, pi)

    Noise is complex AWGN: n with mean = 0 and variance = (No/(2dt)), with No = Eb / SNR_linear.

    Args:
        transmitted_signal (array): Modulated continuous-time signal.
        L_pulse (int)             : Samples per symbol.
        M (int)                   : Modulation order (e.g. 2 for BPSK).
        SNR_dB (float)            : Eb/N0 in dB.
        Eb (float)                : Energy per bit.
        dt (float)                : Sampling interval.

    Returns:
        received_signal (array)        : Faded and noisy received signal.
        rayleigh_coeff_waveform (array): Fading coefficients h(t), constant per symbol.
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