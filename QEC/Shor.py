"""
Shor 9-qubit code: protection of ONE logical qubit against Pauli noise.

The Shor code concatenates the phase-flip and the bit-flip repetition codes:

    |0>_L = (|000> + |111>)(|000> + |111>)(|000> + |111>) / 2^(3/2)
    |1>_L = (|000> - |111>)(|000> - |111>)(|000> - |111>) / 2^(3/2)

    - inside each block of 3 qubits, the parities Z Z detect bit flips (X errors)
    - between the blocks, the parities X^6 detect phase flips (Z errors)
    - Y = iXZ is both a bit flip and a phase flip
    => ANY single-qubit Pauli error (X, Y or Z on any of the 9 qubits) is corrected.

Noise model: every physical qubit independently suffers X, Y or Z with probabilities
px, py, pz (Pauli channel). The code fails only when 2 or more errors hit in a way it
cannot distinguish, so the logical error scales as p^2 instead of p.

Contents, in order:
    1. create_state         -> 1-qubit state with parameters (theta, phi)
    2. shor_code_circuit    -> encoding, noise window, syndromes, correction, decoding
    3. measure_state        -> measurement of the logical qubit (X, Y, Z or "state" basis)
    + test: comparison between the state with and without QEC
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================================
# 1. Single-qubit state
# =====================================================================================

def create_state(theta, phi):
    """ Prepare |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>,
    i.e. the point (theta, phi) of the Bloch sphere.
    For a generic (theta, phi) this is NOT a stabilizer state: it is not an eigenstate
    of X, Y or Z, so all three measurement bases carry information.
    params:
        theta: polar angle on the Bloch sphere (0 = |0>, pi = |1>)
        phi: azimuthal angle (relative phase)
    """
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


# =====================================================================================
# Noise model
# =====================================================================================

def pauli_noise_model(px=0.005, py=0.005, pz=0.005):
    """ Pauli channel on every 'id' gate (the noise window), independently on each qubit:
        rho -> (1 - px - py - pz) rho + px X rho X + py Y rho Y + pz Z rho Z
    All the other gates are ideal, so the comparison only measures the effect of the
    noise window, with and without QEC.
    params:
        px, py, pz: probabilities of an X, Y, Z error on each qubit
    """
    error = pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', 1 - px - py - pz)])
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(error, ['id'])
    return noise


# =====================================================================================
# 2. Shor code
# =====================================================================================

# Syndrome (2 parities, read as an integer) -> index (0, 1, 2) of the element to correct.
#   bit 0 = parity of elements (0, 1), bit 1 = parity of elements (1, 2)
#   error on 0 -> 01 = 1,   on 1 -> 11 = 3,   on 2 -> 10 = 2,   no error -> 00 = 0
# Used both for the qubits inside a block (bit flips) and for the 3 blocks (phase flips).
SYNDROME_TABLE = {1: 0, 3: 1, 2: 2}


def _shor_encode(qc, q):
    """ Encoding of the state on q[0] into the 9 qubits q[0..8]. """
    qc.cx(q[0], q[3])                    # phase-flip code: 3 blocks with leaders 0, 3, 6
    qc.cx(q[0], q[6])
    qc.h([q[0], q[3], q[6]])
    for lead in (0, 3, 6):               # bit-flip code inside each block
        qc.cx(q[lead], q[lead + 1])
        qc.cx(q[lead], q[lead + 2])


def _shor_decode(qc, q):
    """ Inverse of _shor_encode: the logical qubit goes back to q[0]. """
    for lead in (0, 3, 6):
        qc.cx(q[lead], q[lead + 2])
        qc.cx(q[lead], q[lead + 1])
    qc.h([q[0], q[3], q[6]])
    qc.cx(q[0], q[6])
    qc.cx(q[0], q[3])


def shor_code_circuit(state_circ, correct=True):
    """ Protect the 1-qubit state prepared by state_circ with the Shor code.

    Steps:
        1. prepare the state on data[0] (state_circ)
        2. encode it into 9 qubits
        3. noise window: 'id' on the 9 data qubits
        4. bit-flip correction: for each block, parities Z0Z1 and Z1Z2 (2 ancillas),
           then X on the qubit indicated by the syndrome
        5. phase-flip correction: parities X^6 of blocks (0,1) and (1,2), then Z on the
           block indicated by the syndrome (a Z on any qubit of the block is enough)
        6. decode: the corrected logical qubit is back on data[0]

    A Y error is seen by both syndromes: step 4 removes its X part, step 5 its Z part.
    The 2 ancillas are reset and reused for every syndrome: 9 + 2 = 11 qubits in total.
    The circuit is NOT measured: use measure_state(circ, ..., qubit=0).
    params:
        state_circ: 1-qubit QuantumCircuit that prepares the state
        correct: if False, the syndromes are measured but no correction is applied
    """
    data = QuantumRegister(9, 'data')
    anc = QuantumRegister(2, 'anc')
    syn_bit = [ClassicalRegister(2, f'synX{k}') for k in range(3)]   # bit-flip syndromes
    syn_phase = ClassicalRegister(2, 'synZ')                          # phase-flip syndrome
    qc = QuantumCircuit(data, anc, *syn_bit, syn_phase)
    blocks = [data[3*k: 3*k + 3] for k in range(3)]

    # 1-2. state preparation and encoding
    qc.compose(state_circ, qubits=[data[0]], inplace=True)
    _shor_encode(qc, data)
    qc.barrier()

    # 3. noise window
    qc.id(data)
    qc.barrier()

    # 4. bit-flip correction, block by block
    for b, s in zip(blocks, syn_bit):
        qc.cx(b[0], anc[0]); qc.cx(b[1], anc[0])         # anc[0] = Z0 Z1 parity
        qc.cx(b[1], anc[1]); qc.cx(b[2], anc[1])         # anc[1] = Z1 Z2 parity
        qc.measure(anc, s)
        if correct:
            for value, target in SYNDROME_TABLE.items():
                with qc.if_test((s, value)):
                    qc.x(b[target])
        qc.reset(anc)
    qc.barrier()

    # 5. phase-flip correction: X-parities measured with ancillas in |+>
    #    (CNOT with the ANCILLA as control copies the X-parity of the targets onto it)
    qc.h(anc)
    for a, pair in zip(anc, ((0, 1), (1, 2))):
        for k in pair:
            for q in blocks[k]:
                qc.cx(a, q)
    qc.h(anc)
    qc.measure(anc, syn_phase)
    if correct:
        for value, target in SYNDROME_TABLE.items():
            with qc.if_test((syn_phase, value)):
                qc.z(blocks[target][0])
    qc.reset(anc)
    qc.barrier()

    # 6. decoding
    _shor_decode(qc, data)
    return qc


def unprotected_circuit(state_circ):
    """ Reference without QEC: the same state and the same noise window on ONE qubit. """
    qc = state_circ.copy()
    qc.barrier()
    qc.id(0)
    return qc


# =====================================================================================
# 3. Measurement
# =====================================================================================

def measure_state(circ, noise_model=None, basis='Z', qubit=0, shots=4000, state_circ=None):
    """ Measure one qubit of circ and return its counts {'0': n0, '1': n1}.
        'Z'     : computational basis
        'X'     : H before measuring          ('0' = |+>,  '1' = |->)
        'Y'     : Sdg + H before measuring    ('0' = |+i>, '1' = |-i>)
        'state' : undo state_circ before measuring: '0' means "the qubit is in |psi>",
                  so P('0') is directly the FIDELITY with the ideal state
    The other classical registers of circ (syndromes) are dropped from the counts.
    params:
        circ: QuantumCircuit without final measurements
        noise_model: NoiseModel for AerSimulator, None for an ideal simulation
        basis: 'X', 'Y', 'Z' or 'state'
        qubit: index of the qubit to measure (the logical qubit is 0 in both circuits)
        shots: number of measurement shots
        state_circ: circuit that prepared the state (only for basis='state')
    """
    qc = circ.copy()
    out = ClassicalRegister(1, 'meas')
    qc.add_register(out)
    if basis == 'X':
        qc.h(qubit)
    elif basis == 'Y':
        qc.sdg(qubit)
        qc.h(qubit)
    elif basis == 'state':
        qc.compose(state_circ.inverse(), qubits=[qubit], inplace=True)
    qc.measure(qubit, out[0])

    # no transpile: it would remove the 'id' gates, and with them the noise.
    # shot branching: shots with the same noise/measurement history are simulated together
    sim = AerSimulator(noise_model=noise_model, shot_branching_enable=True)
    raw = sim.run(qc, shots=shots).result().get_counts()

    counts = {'0': 0, '1': 0}
    for key, n in raw.items():
        counts[key.split()[0]] += n        # 'meas' is the last register added -> leftmost
    return counts


def bloch_vector(counts_by_basis):
    """ Bloch vector (<X>, <Y>, <Z>) from the counts in the three bases: <P> = P(0) - P(1). """
    return np.array([(c['0'] - c['1']) / (c['0'] + c['1'])
                     for c in (counts_by_basis[b] for b in 'XYZ')])


# =====================================================================================
# Test
# =====================================================================================

if __name__ == "__main__":
    # --- parameters
    theta, phi = np.pi / 5, np.pi / 10      # generic point of the Bloch sphere (not a stabilizer state)
    px = 0.01
    py = 0
    pz = 0                                 # Pauli error probabilities per physical qubit
    shots = 100_000                        # per measurement: infidelities are ~1e-2 or less

    state = create_state(theta, phi)
    noise = pauli_noise_model(px, py, pz)
    psi = Statevector(state)
    r_ideal = np.array([psi.expectation_value(Pauli(P)).real for P in 'XYZ'])
    print(f"State: theta = {theta:.3f}, phi = {phi:.3f}")
    print(psi)
    print(f"Exact Bloch vector <X>, <Y>, <Z> = {np.round(r_ideal, 4)}")
    print(f"Pauli noise per physical qubit: px = {px}, py = {py}, pz = {pz}")
    print(f"Shots per measurement: {shots}\n")

    cases = {
        'Ideal': (unprotected_circuit(state), None),
        'No QEC': (unprotected_circuit(state), noise),
        'Shor QEC': (shor_code_circuit(state), noise),
    }

    # --- measurements: three Pauli bases (histograms) + projection on |psi> (fidelity)
    results, infid, infid_err = {}, {}, {}
    for label, (circ, nm) in cases.items():
        results[label] = {b: measure_state(circ, nm, b, shots=shots) for b in 'XYZ'}
        c = measure_state(circ, nm, 'state', shots=shots, state_circ=state)
        F = c['0'] / shots
        infid[label] = 1 - F
        infid_err[label] = np.sqrt(F * (1 - F) / shots)      # binomial standard error

    # --- comparison
    print(f"{'':10s}{'P0 (X)':>9s}{'P0 (Y)':>9s}{'P0 (Z)':>9s}{'<X>':>9s}{'<Y>':>9s}{'<Z>':>9s}"
          f"{'1 - F':>10s}{'+/-':>9s}")
    for label, res in results.items():
        p0 = [res[b]['0'] / shots for b in 'XYZ']
        r = bloch_vector(res)
        print(f"{label:10s}" + ''.join(f"{v:9.4f}" for v in (*p0, *r))
              + f"{infid[label]:10.4f}{infid_err[label]:9.4f}")
    print(f"\nInfidelity: no QEC = {infid['No QEC']:.4f} +/- {infid_err['No QEC']:.4f},  "
          f"Shor QEC = {infid['Shor QEC']:.4f} +/- {infid_err['Shor QEC']:.4f}  "
          f"-> reduced by a factor {infid['No QEC'] / infid['Shor QEC']:.1f}")

    # --- figure: one histogram per basis (three cases side by side) + infidelity
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    width = 0.27
    for ax, b in zip(axes[:3], 'XYZ'):
        for i, label in enumerate(results):
            probs = [results[label][b][o] / shots for o in '01']
            bars = ax.bar(np.arange(2) + (i - 1) * width, probs, width, label=label)
            ax.bar_label(bars, fmt='%.3f', fontsize=7)
        ax.set_xticks([0, 1], ['0', '1'])
        ax.set_ylim(0, 1)
        ax.set_title(f'{b} basis')
        ax.set_xlabel('Outcome')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Probability')
    axes[0].legend(loc='upper right')

    ax = axes[3]
    labels = ['No QEC', 'Shor QEC']
    ax.bar(labels, [infid[l] for l in labels], yerr=[infid_err[l] for l in labels],
           capsize=6, color=['C1', 'C2'])
    ax.set_title('Infidelity  1 - F')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'State (theta, phi) = ({theta:.2f}, {phi:.2f}),  Pauli noise per qubit: '
                 f'px = {px}, py = {py}, pz = {pz}')
    fig.tight_layout()

    fig.savefig('shor_comparison.png', dpi=150, bbox_inches='tight')
    print("Figure saved as shor_comparison.png")
    # plt.show()   # uncomment if interactive windows work on your system