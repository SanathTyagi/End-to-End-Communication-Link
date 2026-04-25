import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from Rayleigh import Rayleigh_Channel
from Equalizer import Equalizer
from AWGN_Channel import AWGN_Channel


def ber_simulation(encoded_bits,coded_bits,Monte_Carlo_Runs,p,M,Eb,dt):
    L_pulse=len(p)
    #each run comprises of size 1000
    Batch_size=1000
    num_batches=Monte_Carlo_Runs//Batch_size

    
    SNR_dB_values = np.arange(0,11,1)
    num_points = len(SNR_dB_values)
    BER_theory_Rayleigh = np.zeros(num_points)
    BER_theory_BPSK_AWGN = np.zeros(num_points)
    BER_Rayleigh = np.zeros(num_points)
    BER_AWGN = np.zeros(num_points)
    for i,SNR_dB in enumerate(SNR_dB_values):
        print(f"--- Simulating SNR = {SNR_dB} dB ---")
        err_Rayleigh=0
        err_AWGN=0
        for j in range(num_batches):
            transmitted_bits=np.tile(coded_bits,Batch_size)
            transmitted_symbols=np.sqrt(Eb)*(1-(2*transmitted_bits))

            #=========================
            #Rayleigh BER
            #=========================
            received_signal_Rayleigh, h = Rayleigh_Channel(transmitted_symbols, 1, M, SNR_dB, Eb, 1)
            received_signal_Rayleigh_equalised = Equalizer(received_signal_Rayleigh, h, SNR_dB, Eb)
            #Decision Rule
            estimated_bits_Rayleigh=np.where(np.real(received_signal_Rayleigh_equalised)>=0, 0, 1)
            #BER for Rayleigh
            estimated_bits_matrix_Rayleigh = np.array(estimated_bits_Rayleigh).reshape(Batch_size,-1)
            err_Rayleigh += np.sum(estimated_bits_matrix_Rayleigh!=coded_bits)

            #=========================
            #AWGN BER
            #=========================
            received_signal = AWGN_Channel(transmitted_symbols,SNR_dB,Eb,1)
            #Decision Rule
            estimated_bits_AWGN=np.where(received_signal>=0, 0, 1)
            
            #BER for AWGN
            estimated_bits_matrix_AWGN = np.array(estimated_bits_AWGN).reshape(Batch_size,-1)
            err_AWGN += np.sum(estimated_bits_matrix_AWGN!=coded_bits)
        
        BER_Rayleigh[i] = err_Rayleigh/(len(coded_bits)*Monte_Carlo_Runs)
        BER_AWGN[i] = err_AWGN/(len(coded_bits)*Monte_Carlo_Runs)
        
        #theoretical_BER
        SNR_linear = 10**(SNR_dB/10)
        No = Eb/SNR_linear
        BER_theory_Rayleigh[i] = (1-np.sqrt(SNR_linear/(1+SNR_linear)))/2
        BER_theory_BPSK_AWGN[i] = 0.5*erfc(np.sqrt(SNR_linear)) 

    plt.figure(figsize=(10,6),dpi=150)
    plt.ylim(1e-5,1)
    plt.xlim(0,10)
    plt.semilogy(SNR_dB_values,BER_theory_Rayleigh,label='Theoretical Rayleigh BER',color='blue',linewidth=2)
    plt.semilogy(SNR_dB_values,BER_Rayleigh,label='Simulated Rayleigh BER',color='orange',marker='o',linestyle='--',markersize=5)
    plt.semilogy(SNR_dB_values,BER_theory_BPSK_AWGN,label='Theoretical AWGN BER',color='green',marker='s',linewidth=2,markersize=5)
    plt.semilogy(SNR_dB_values,BER_AWGN,label='Simulated AWGN BER',color='red',marker='^',linestyle='--',markersize=5)
    
    plt.title('Bit Error Rate Rayleigh vs AWGN', fontsize=14, fontweight='bold')
    plt.xlabel('SNR ($E_b/N_0$) [dB]', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.show()
    return




