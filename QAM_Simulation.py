import numpy as np
from AWGN_Channel import AWGN_Channel
from scipy.special import erfc
import matplotlib.pyplot as plt
from Convolutional_Coding import Convolutional_Coding
from QAM import PAM_table, QAM_table, QAM_map, Nearest_Neighbour




def QAM_AWGN(coded_bits, M, Eb,SNR_dB):
    '''Args:
    coded_bits: either after channel coding (convolutional coding) or directly after source coding.
    M: modulation order. Limited to even QAMs in this simulation.
    Eb: energy per bit (although Es is set to 1 during the SNR estimation, Eb is given as input, as done in BPSK).
    SNR_dB: signal-to-noise ratio in dB.
    Working:
    1. Reshape coded_bits into 2-D.
    2. Decide the amplitude level A.
    3. Make symbols with QAM_map (Gray coding).
    4. Add AWGN.
    5. Nearest-neighbour estimation for each received complex value r.
    Returns:
    received_coded_bits_2d.
    '''
    # bits per symbol
    k=int(np.log2(M))
    # bits per I/Q phase
    m=k//2
    
    # relation between Eb and Es
    Es=Eb*k
    # reshaping coded bits
    k=int(np.log2(M))
    coded_bits_2d=np.reshape(coded_bits,(-1,k))

    # amplitude level
    A=np.sqrt(3*Es/(2*(M-1)))

    # transmitted symbols
    symbols = QAM_map(coded_bits_2d, M,A)

    # AWGN noise
    received_signal = AWGN_Channel(symbols,SNR_dB,Eb,1)


    # receiver
    # received_signal = I+jQ
    # thinking of making a dict whose key will contain the constellation bits, and whose value would contain
    # |r-(I_org+Q_org)|
    # finally pick the value by argmin.
    qam_dict=QAM_table(m, A)
    received_keys = []                      
    for r in received_signal:               
        key = Nearest_Neighbour(qam_dict, r)
        received_keys.append(key)
    
    rows=[]
    for s in received_keys:
        rows.append(list(s))
    received_coded_bits_2d = np.array(rows, dtype=int)
    return received_coded_bits_2d


                       
def Monte_Carlo_QAM_AWGN(M, SNR_dB_values, threshold_errors=300, batch_size=2400, max_bits=10_000_000):
    '''Args:
    threshold_errors: at each SNR point, bits are evaluated until we cross the threshold errors, which is set to 300.
    batch_size: in a single batch we process 2400 bits so that we can handle 16, 64, 256, 1024 QAM simultaneously.
    Max_bits: to ensure that the simulation does not fall into an endless run.
    Working:
    For each SNR_dB point:
     Evaluate errors by running the batch once, then check whether the errors passed the threshold.
     Once they pass the threshold, record the BER and store it in BER_AWGN, along with the theory value.
    Returns:
    BER_AWGN: array of BER at each SNR_dB point, in order.
    '''
    k = int(np.log2(M))
    Eb = 1/k                                   # Es = k*Eb = 1
    num_points = len(SNR_dB_values)
    BER_AWGN = np.zeros(num_points)
    BER_AWGN_theory = np.zeros(num_points)

    for j, SNR_dB in enumerate(SNR_dB_values):
        errors = 0
        i = 1
        while (i > 0):
            coded_bits = np.random.choice([0, 1], size=batch_size)
            assert len(coded_bits) % k == 0, f"coded_bits length must be a multiple of k={k}"
            coded_bits_2d = coded_bits.reshape(-1, k)
            length = i*batch_size
            received_coded_bits_2d = QAM_AWGN(coded_bits, M, Eb, SNR_dB)
            errors += np.sum(coded_bits_2d != received_coded_bits_2d)
            if errors >= threshold_errors or length >= max_bits:
                break
            i += 1
        BER = errors/length
        BER_AWGN[j] = BER
        SNR_linear = 10**(SNR_dB/10)
        x = np.sqrt(3*k*SNR_linear/(M-1))
        BER_AWGN_theory[j] = (4/k)*(1 - 1/np.sqrt(M)) * 0.5*erfc(x/np.sqrt(2))

    return BER_AWGN, BER_AWGN_theory


# Run and plot
np.random.seed(0)
SNR_dB_values = np.arange(0, 16, 1)   # SNR from 0 to 15 dB   

plt.figure(figsize=(10, 6), dpi=150)
for M, color in [(16, 'tab:blue'), (64, 'tab:green'), (256, 'tab:orange'), (1024, 'tab:red')]:
    sim, theory = Monte_Carlo_QAM_AWGN(M, SNR_dB_values)
    plt.semilogy(SNR_dB_values, sim, 'o-', color=color, label=f'{M}-QAM Simulation')
    plt.semilogy(SNR_dB_values, theory, '--', color=color, label=f'{M}-QAM Theory (Gray approx.)')

plt.title('Bit Error Rate: 16,64,256,1024-QAM over AWGN.', fontsize=14, fontweight='bold')
plt.xlabel('SNR ($E_b/N_0$) [dB]', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='best')
plt.savefig('QAM_AWGN_BER.png', dpi=150, bbox_inches='tight')
plt.show()