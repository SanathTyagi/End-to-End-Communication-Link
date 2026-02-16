import numpy as np
from Source_Coding import get_huffman_bits
from Source_Coding import decode_huffman_bits
from BPSK_Modulator import BPSK_modulator
from AWGN_Channel import AWGN_Channel
from Convolutional_Coding import Convolutional_Coding
from Demodulator_BPSK import Demodulator_BPSK
from Viterbi_Decoder import Viterbi_Decoder
from Simulation import ber_simulation
# ==========================================
# 1. SYSTEM CONFIGURATIONS (The "Knobs")
# ==========================================
fs=1000          # Sampling Frequency (Hz)
dt=1/fs          # Time Step
T=2              # Symbol Duration (Seconds)
A=1              # Signal Amplitude
fc=2/T           # Carrier Frequency (for the basis function)
Monte_Carlo_Runs=10000 # Total bits to simulate for BER curves



# ==========================================
# 2. PULSE SHAPE GENERATION
# ==========================================
# Defining the basis function (Unit Energy Pulse)
t_pulse=np.arange(0,T,dt)
def pulse(T,fc,t_pulse):
        return np.sqrt(2/T)*np.cos(2*np.pi*fc*t_pulse)
p=pulse(T,fc,t_pulse)
Ep=np.sum(p*p)*dt  # Energy of the pulse

# ==========================================
# 3. THE MAIN PIPELINE 
# ==========================================
print("--- Starting System Validation ---")
input_text="sanath"
print(f"Original Message: {input_text}")

# Step A: Source Coding
encoded_bits,huffman_root=get_huffman_bits(input_text)

# Step B: Channel Coding (Adding Redundancy)
coded_bits=Convolutional_Coding(encoded_bits)

# Step C: Modulation (Generating the Waveform)
t,transmitted_signal,M=BPSK_modulator(coded_bits,dt,T,A,p)
Eb=Ep/np.log2(M) # For BPSK, Energy per bit = Energy per pulse

# Step D: Ideal Demodulation & Decoding (Checking if the system is perfect without noise)
soft_bits_ideal,estimated_bits_ideal = Demodulator_BPSK(transmitted_signal, p, dt)
decoded_bits_ideal=Viterbi_Decoder(estimated_bits_ideal)

# Reshape and remove flush bits to reconstruct original text
decoded_bits_matrix=np.array(decoded_bits_ideal).reshape(-1, len(encoded_bits)+2)
clean_bits=decoded_bits_matrix[0, :-2].tolist()
reconstructed_text=decode_huffman_bits(clean_bits, huffman_root)
print(f"Reconstructed Message: {reconstructed_text}")
print("System Validation: SUCCESS\n")


# ==========================================
# 4. PERFORMANCE ANALYSIS (Monte Carlo)
# ==========================================
print(f"--- Running Monte Carlo Simulation ({Monte_Carlo_Runs} bits) ---")
#ber_simulation function will handle the loop, the AWGN channel, and plot the final BER curves.
ber_simulation(encoded_bits,coded_bits,Monte_Carlo_Runs,transmitted_signal,p,Eb,dt)

print("Simulation Complete. BER Plot and Constellation saved.")