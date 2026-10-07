#create N bell pairs on a 2N qubit system

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer import AerSimulator
from qiskit import transpile

def create_bell_state(N):
    qc = QuantumCircuit(2*N)
    for i in range(N):
        qc.h(2*i)
        qc.cx(2*i, 2*i+1)
    return qc


def print_circuit(circ):
    print(circ.draw())


def measure_and_plot(circ, shots=1024):
    ''' Measure all qubits, print the counts, and display the histogram. No noise is applied.
        params: 
            circ: QuantumCircuit to be measured
            shots: number of measurement shots
    
            THE PLOT IS NOT AUTOMATICALLY SHOWN; CALL plt.show() AFTERWARDS IF DESIRED.
        '''
    # copy of the circuit with measurements (the original circuit remains intact)
    qc_meas = circ.measure_all(inplace=False)

    sampler = StatevectorSampler()
    result = sampler.run([qc_meas], shots=shots).result()[0]
    counts = result.data.meas.get_counts()   # 'meas' is the register created by measure_all

    print(counts)
    plot_histogram(counts)
    return counts

def noisy_measure_and_plot(circ, p1=0.01, p2=0.03, p_ro=0.02, shots=1024):
    ''' Create a noisy model with single and two-qubit depolarizing errors and readout error and measure the circuit.
    params: 
        circ: QuantumCircuit to be measured
        p1: single-qubit depolarizing error probability
        p2: two-qubit depolarizing error probability
        p_ro: readout error probability
        shots: number of measurement shots

        THE PLOT IS NOT AUTOMATICALLY SHOWN; CALL plt.show() AFTERWARDS IF DESIRED.
        
    '''
    noise = NoiseModel()
    # error  depolarizing on 1 and 2 qubit gates
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['h', 'x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    # readout error: P(read 1 | is 0) = p_ro, P(read 0 | is 1) = p_ro
    noise.add_all_qubit_readout_error(ReadoutError([[1-p_ro, p_ro], [p_ro, 1-p_ro]]))

    sim = AerSimulator(noise_model=noise)
    qc_meas = transpile(circ.measure_all(inplace=False), sim)
    counts = sim.run(qc_meas, shots=shots).result().get_counts()

    print(counts)
    plot_histogram(counts)
    return counts 


## Test

if __name__ == "__main__":
    N = 2  # Number of Bell pairs, the number of qubits will be 2*N
    qc = create_bell_state(N)
    Bell_state = Statevector(qc)
    print(Bell_state)
    print_circuit(qc)

    counts = measure_and_plot(qc, shots=2048)
    counts_noisy = noisy_measure_and_plot(qc, p1=0.01, p2=0.03, p_ro=0.02, shots=2048) 

    plt.show()