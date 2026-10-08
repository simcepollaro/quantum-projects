#create N bell pairs on a 2N qubit system
#
# + bit-flip quantum error correction (3-qubit repetition code) for a GENERIC circuit:
#   the QEC functions take any QuantumCircuit, extract its statevector and number of
#   qubits, and protect every qubit against X errors. The Bell state is just the test case.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, pauli_error
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


# =====================================================================================
# Bit-flip noise and bit-flip QEC
# =====================================================================================

def bit_flip_noise_model(p):
    ''' Noise model with ONLY X errors: every 'id' gate flips its qubit with probability p.
    All other gates are ideal, so the 'id' gates act as the noise window
    (think of the qubits waiting idle while errors accumulate).
    params:
        p: bit-flip probability per qubit
    '''
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(pauli_error([('X', p), ('I', 1 - p)]), ['id'])
    return noise


def run_bit_flip(circ, p, shots=1024):
    ''' Counts of circ after the bit-flip noise window, without QEC (no plot). '''
    qc = circ.copy()
    qc.barrier()
    qc.id(range(qc.num_qubits))          # noise window: X with probability p on each qubit
    qc.measure_all()
    sim = AerSimulator(noise_model=bit_flip_noise_model(p))
    return sim.run(qc, shots=shots).result().get_counts()


def bit_flip_measure_and_plot(circ, p, shots=1024):
    ''' Prepare the state of circ, apply the bit-flip noise window to every qubit and
    measure. NO error correction: this is the reference to compare with.
    params:
        circ: QuantumCircuit that prepares the state
        p: bit-flip probability per qubit
        shots: number of measurement shots

        THE PLOT IS NOT AUTOMATICALLY SHOWN; CALL plt.show() AFTERWARDS IF DESIRED.
    '''
    counts = run_bit_flip(circ, p, shots)
    print(counts)
    plot_histogram(counts, title=f'No QEC, bit-flip p = {p}')
    return counts


# Syndrome of one 3-qubit block -> qubit of the block to flip.
# Syndrome bit 0 = parity(q0, q1), syndrome bit 1 = parity(q1, q2), read as an integer:
#   X on q0 -> 01 = 1,   X on q1 -> 11 = 3,   X on q2 -> 10 = 2,   no error -> 00 = 0
SYNDROME_TABLE = {1: 0, 3: 1, 2: 2}


def bit_flip_qec_circuit(circ, correct=True):
    ''' Build the bit-flip QEC circuit for the GENERIC state prepared by circ.

    Everything is extracted from circ:
        n   = number of qubits of circ (logical qubits)
        psi = Statevector(circ), the state to protect

    Mapping: each logical qubit k is encoded in the 3 data qubits (3k, 3k+1, 3k+2):
        a|0> + b|1>  ->  a|000> + b|111>
    so a generic n-qubit state  sum_x c_x |x>  becomes  sum_x c_x |xxx...>  on 3n qubits.

    Steps:
        1. prepare psi on the first qubit of each block (data[0], data[3], data[6], ...)
        2. encode every block with two CNOTs
        3. noise window: 'id' on all 3n data qubits
        4. for every block: measure the 2 parities with 2 ancillas (syndrome), apply the
           correction if correct=True, reset the ancillas so the next block can reuse them
        5. decode every block (inverse of step 2): logical qubit k is back on data[3k]
        6. measure data[3k] for every k -> n bits, the same format as the unprotected circuit

    Qubits used: 3n data + 2 ancillas (shared by all blocks).
    params:
        circ: QuantumCircuit that prepares the state to protect (no measurements)
        correct: if False, the syndrome is measured but no correction is applied
    '''
    psi = Statevector(circ)
    n = circ.num_qubits

    data = QuantumRegister(3 * n, 'data')
    anc = QuantumRegister(2, 'anc')
    syn = [ClassicalRegister(2, f'syn{k}') for k in range(n)]   # one syndrome per block
    out = ClassicalRegister(n, 'out')
    qc = QuantumCircuit(data, anc, *syn, out)

    blocks = [data[3*k: 3*k + 3] for k in range(n)]
    logical = [b[0] for b in blocks]          # qubit that carries logical qubit k

    # 1. state preparation: psi on the logical carriers (qubit k of psi -> data[3k])
    qc.initialize(psi.data, logical)

    # 2. encoding
    for b in blocks:
        qc.cx(b[0], b[1])
        qc.cx(b[0], b[2])
    qc.barrier()

    # 3. noise window
    qc.id(data)
    qc.barrier()

    # 4. syndrome measurement + correction, block by block
    for b, s in zip(blocks, syn):
        qc.cx(b[0], anc[0]); qc.cx(b[1], anc[0])     # anc[0] = q0 XOR q1
        qc.cx(b[1], anc[1]); qc.cx(b[2], anc[1])     # anc[1] = q1 XOR q2
        qc.measure(anc, s)
        if correct:
            for value, target in SYNDROME_TABLE.items():
                with qc.if_test((s, value)):
                    qc.x(b[target])
        qc.reset(anc)                                 # ancillas back to |0> for the next block
    qc.barrier()

    # 5. decoding
    for b in blocks:
        qc.cx(b[0], b[2])
        qc.cx(b[0], b[1])
    qc.barrier()

    # 6. measurement of the logical qubits
    qc.measure(logical, out)
    return qc


def run_bit_flip_qec(circ, p, shots=1024, correct=True):
    ''' Counts of the n logical qubits of the QEC circuit of circ (no plot).
    The syndrome bits are dropped, so the keys are the same as the unprotected circuit.
    '''
    qc = bit_flip_qec_circuit(circ, correct)
    # shot branching: shots that follow the same noise/measurement trajectory are simulated
    # together instead of one by one (same result, ~4x faster for this circuit)
    sim = AerSimulator(noise_model=bit_flip_noise_model(p), shot_branching_enable=True)
    # no transpile here: it would remove the 'id' gates, and with them the noise
    raw = sim.run(qc, shots=shots).result().get_counts()

    counts = {}
    for key, cnt in raw.items():
        logical = key.split()[0]       # keys are 'out syn(n-1) ... syn0': last register is leftmost
        counts[logical] = counts.get(logical, 0) + cnt
    return counts


def bit_flip_qec_measure_and_plot(circ, p, shots=1024, correct=True):
    ''' Run the bit-flip QEC circuit of circ under bit-flip noise and return the counts
    of the n logical qubits (syndrome bits are dropped), with the same keys as the
    unprotected circuit.
    params:
        circ: QuantumCircuit that prepares the state to protect
        p: bit-flip probability per qubit
        shots: number of measurement shots
        correct: apply the correction (True) or only measure the syndrome (False)

        THE PLOT IS NOT AUTOMATICALLY SHOWN; CALL plt.show() AFTERWARDS IF DESIRED.
    '''
    counts = run_bit_flip_qec(circ, p, shots, correct)
    print(counts)
    label = 'QEC' if correct else 'encoded, no correction'
    plot_histogram(counts, title=f'{label}, bit-flip p = {p}')
    return counts


# =====================================================================================
# Comparison QEC vs no QEC
# =====================================================================================

def wrong_outcome_rate(circ, counts):
    ''' Fraction of shots giving an outcome that the ideal state of circ can never produce.
    Generic error measure; for Bell pairs it is the probability of 01 or 10 in a pair.
    '''
    ideal = Statevector(circ).probabilities_dict()
    total = sum(counts.values())
    return sum(c for k, c in counts.items() if ideal.get(k, 0) < 1e-12) / total


def compare_qec(circ, p_values, shots=2000):
    ''' Error of the state prepared by circ with and without bit-flip QEC, for every p.
    Two figures of merit:
        - wrong outcome rate (lower is better)
        - Hellinger fidelity between measured and ideal distribution (higher is better)
    THE PLOT IS NOT AUTOMATICALLY SHOWN; CALL plt.show() AFTERWARDS IF DESIRED.
    '''
    ideal = Statevector(circ).probabilities_dict()
    sim_runs = {'No QEC': [], 'QEC': []}
    for p in p_values:
        for label, run in (('No QEC', run_bit_flip), ('QEC', run_bit_flip_qec)):
            counts = run(circ, p, shots)
            sim_runs[label].append((wrong_outcome_rate(circ, counts),
                                    hellinger_fidelity(ideal, counts)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, marker in (('No QEC', 'o-'), ('QEC', 's-')):
        err, fid = zip(*sim_runs[label])
        ax1.plot(p_values, err, marker, label=label)
        ax2.plot(p_values, fid, marker, label=label)
    ax1.set_ylabel('Wrong outcome rate')
    ax2.set_ylabel('Hellinger fidelity')
    for ax in (ax1, ax2):
        ax.set_xlabel('Bit-flip probability p per qubit')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'Bit-flip QEC vs no QEC ({circ.num_qubits} logical qubits)')
    fig.tight_layout()
    return sim_runs


## Test

if __name__ == "__main__":
    N = 2  # Number of Bell pairs, the number of qubits will be 2*N
    qc = create_bell_state(N)
    Bell_state = Statevector(qc)
    print(Bell_state)
    print_circuit(qc)

    counts = measure_and_plot(qc, shots=2048)

    # --- bit-flip noise: without and with QEC
    p = 0.1
    print(f"\nBit-flip noise p = {p}")
    print("No QEC:")
    counts_noisy = bit_flip_measure_and_plot(qc, p, shots=2048*2)
    print("QEC:")
    counts_qec = bit_flip_qec_measure_and_plot(qc, p, shots=2048*2)

    for label, c in (('No QEC', counts_noisy), ('QEC', counts_qec)):
        print(f"{label:7s} wrong outcome rate = {wrong_outcome_rate(qc, c):.4f}   "
              f"Hellinger fidelity = {hellinger_fidelity(Bell_state.probabilities_dict(), c):.4f}")

    # --- sweep over p
    compare_qec(qc, np.linspace(0, 0.5, 11))

    for n in plt.get_fignums():
        plt.figure(n).savefig(f"figure_{n}.png", dpi=150, bbox_inches="tight")
    print("Figures saved as figure_*.png")