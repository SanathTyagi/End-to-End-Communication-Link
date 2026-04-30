import numpy as np
from Source_Coding import get_huffman_bits
from Convolutional_Coding import Convolutional_Coding
from BPSK_Modulator import BPSK_modulator
from AWGN_Channel import AWGN_Channel
from Rayleigh import Rayleigh_Channel
from Equalizer import Equalizer
from Demodulator_BPSK import Demodulator_BPSK
from Viterbi_Decoder import Viterbi_Decoder
from Source_Coding import decode_huffman_bits
from Simulation import ber_simulation
from Simulation import plot_constellation_BPSK
# ==========================================
# 1. SYSTEM CONFIGURATIONS (The "Knobs")
# ==========================================
fs=1000          # Sampling Frequency (Hz)
dt=1/fs          # Time Step
T=2              # Symbol Duration (Seconds)
A=1              # Signal Amplitude
fc=2/T           # Carrier Frequency (for the basis function)
Monte_Carlo_Runs=100000 # Total bits to simulate for BER curves
SNR_dB=3



# ==========================================
# 2. PULSE SHAPE GENERATION
# ==========================================
# Defining the basis function (Unit Energy Pulse)
t_pulse=np.arange(0,T,dt)
def pulse(T,fc,t_pulse):
        return np.sqrt(2/T)*np.cos(2*np.pi*fc*t_pulse)
p=pulse(T,fc,t_pulse)
L_pulse=len(p)
Ep=np.sum(p*p)*dt  # Energy of the pulse


# ==========================================
# 3. THE MAIN PIPELINE - Pulse Level
# ==========================================
print("--- Starting System Validation ---")
input_text="End-to-End PHY Layer Simulation by Sanath Tyagi"
print(f"Original Message: {input_text}")

# Step A: Source Coding
encoded_bits, huffman_root = get_huffman_bits(input_text)

# Step B: Channel Coding (Adding Redundancy)
coded_bits = Convolutional_Coding(encoded_bits)

# Expanding bit sequence specifically for dense constellation plotting
constellation_coded_bits = np.tile(coded_bits, 10)

# Step C: Modulation (Generating the Waveform)
t, transmitted_signal, M = BPSK_modulator(constellation_coded_bits, dt, T, A, p)
Eb = Ep / np.log2(M) # For BPSK, Energy per bit = Energy per pulse

# Step D: Channel Modeling (Adding Rayleigh Fading)
received_signal, h = Rayleigh_Channel(transmitted_signal, L_pulse, M, SNR_dB, Eb, dt)

# Step E: Receiver Processing (Equalization & Demodulation)
# Compensating for the channel fading
received_signal_equalized = Equalizer(received_signal, h, SNR_dB, Eb)

# Extracting soft bits from the real part of the equalized signal
soft_bits_I, estimated_bits = Demodulator_BPSK(np.real(received_signal_equalized), p, dt)
soft_bits_Q, _ = Demodulator_BPSK(np.imag(received_signal_equalized), p, dt)
#BPSK is 1D so we don't need the quadrature estimated_bits

# Step F: Visualization
plot_constellation_BPSK(soft_bits_I, soft_bits_Q, constellation_coded_bits, SNR_dB)

# Step G: Decoding
decoded_bits = Viterbi_Decoder(estimated_bits)

# Reshape and remove flush bits to reconstruct original text
decoded_bits_matrix = np.array(decoded_bits).reshape(-1, len(encoded_bits)+2)
clean_bits = decoded_bits_matrix[0, :-2].tolist()
reconstructed_text = decode_huffman_bits(clean_bits, huffman_root)
print(f"Reconstructed Message: {reconstructed_text} at SNR dB: {SNR_dB}")
print("System Validation: SUCCESS\n")

# ==========================================
# 4.PERFORMANCE ANALYSIS (Monte Carlo) 
# ==========================================
print(f"--- Running Monte Carlo Simulation ({Monte_Carlo_Runs} bits) ---")
ber_simulation(encoded_bits,coded_bits,Monte_Carlo_Runs,p,M,Eb,dt)
print("Simulation Complete.")