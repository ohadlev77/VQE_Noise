import csv
import time
import copy

import pandas as pd

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal

class VQE_NoiseExp:
    """
    Implementation of the experiment depicted in the paper - https://arxiv.org/abs/2108.12388 + extensions.
    For a given Hamiltonian matrix decomposed to a weighted sum of Pauli strings,
    The class constructs an experiment object for becnhmarking VQE's resiliency to noise.
    Any `VQE_NoiseExp` object is being constructed with built-in ansatz circuits,
    as they defined in the class attributes below. Using the class' methods it's possible to run
    various forms of experiments, while the data is being saved automatically to CSV files.
    For a complete guid through see - https://github.com/ohadlev77/VQE_Noise/raw/main/paper.pdf.
    """
    
    # Class attributes
    # Defining the 12 ansatz circuits depicted in the paper (appendix) + 20 new ansatzes
    ent_patterns = {'A': ([2,3],[1,2],[0,1],[0,3]),
                    'B': ([0,1],[0,2],[0,3],[1,2],[1,3],[2,3]),
                    'C': ([0,1],[1,2],[2,3]),
                    'D': ([0,1],[1,2]),
                    'E': ([1,2],[2,3]),
                    'F': ([0,1],[2,3])
                }
    
    # The first record is a None record, for the ID numbers to fit with the numbering in the paper (that starts from 1)
    circuits_def = [None,
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cx', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cz', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cz', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cz', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cx', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cx', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cx', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': 'ry', 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cz', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['h','rx'], 'ent_pattern': ent_patterns['B'], 'ent_gates': ['cx' for i in range(6)], 'reps': 1},
                    {'rotation_pattern': ['h','rx'], 'ent_pattern': ent_patterns['B'], 'ent_gates': ['cz' for i in range(6)], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['B'], 'ent_gates': ['cz' for i in range(6)], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['B'], 'ent_gates': ['cx' for i in range(6)], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cx', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cz', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cz', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cz', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cx', 'cx', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cx', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cx', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['A'], 'ent_gates': ['cz', 'cz', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cx', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cz', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cx', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cx', 'cz', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cx', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cz', 'cx', 'cz'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cz', 'cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cz', 'cz', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry','rz'], 'ent_pattern': ent_patterns['C'], 'ent_gates': ['cx', 'cx', 'cx'], 'reps': 2},
                    {'rotation_pattern': ['ry', 'rz'], 'ent_pattern': ent_patterns['D'],'ent_gates': ['cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry', 'rz'], 'ent_pattern': ent_patterns['E'], 'ent_gates': ['cx', 'cx'], 'reps': 1},
                    {'rotation_pattern': ['ry', 'rz'], 'ent_pattern': ent_patterns['F'], 'ent_gates': ['cx', 'cx'], 'reps': 1},
                    ]

    def __init__(self, H, name, NRE=0):
        """
        Intializing the VQE_NoiseExp object includes:
            # Classical exact evaluation of the minimum eigenvalue of the given Hamiltonain H (feasible for small Hamiltonians).
            # Construction of the ansatz objects as defines in the class attributes.
        Args:
            H(PauliSumOp object) - a Hamiltonian object to perform the VQE upon.
            name(str) - a name given for the experiment by the user (data will be saved according to that name).
            NRE(float) - Nuclear Repulsion Energy - If stated, the NRE will be added to the minimum eigenvalues calculated in this class.
                         If the qubit Hamiltonian H is mapped from a fermionic Hamiltonian of a molecule, it can be used.
                         In the paper the NRE is added to all calculated minimum eigenvalues(= ground state energies).
        """
        self.H = H
        self.name = name
        self.n = H.num_qubits
        self.NRE = NRE

        mes = NumPyMinimumEigensolver()
        # NRE is positive, GSE (Ground State Energy) is negative
        self.exact_min_ev = mes.compute_minimum_eigenvalue(self.H).eigenvalue + NRE

        self.build_circuits_ob()

    def build_circuits_ob(self):
        """
        Implementing the class attribute `circuits_def` as `TwoLocal` ansatz objects (into `self.circuits_ob`).
        Saves only the entanglememnt blocks as `QuantumCircuit` objects into `self.ent_parts_only`.
            # If reps > 1, then a concatanation of the entanglement blocks will be created as 1 `QuantumCircuit` object.
        """

        self.circuits_ob = []
        self.ent_parts_only = []
        for c in self.circuits_def:

            if c is None:
                an = None
                ent_qc = None
            else:
                qc = QuantumCircuit(self.n)
                # Implementing gates
                for cgate, c_t in zip(c['ent_gates'], c['ent_pattern']):
                    getattr(qc, cgate)(c_t[0], c_t[1])

                an = TwoLocal(self.n, rotation_blocks=c['rotation_pattern'], entanglement_blocks=qc, reps=c['reps'], insert_barriers=True)
                ent_qc = copy.deepcopy(qc)
                for i in range(c['reps'] - 1):
                    ent_qc.append(qc, qargs=ent_qc.qubits)
            
            # Adding the whole ansatz circuit to `circuits_ob`
            self.circuits_ob.append(an)

            # Adding the entanglements parts only (concatenated in the case of reps > 1) to `ent_parts_only`
            self.ent_parts_only.append(ent_qc)
    
    def circuits_repr(self, circuits_ids=None):
        """
        Displays circuits in `self.circuits_ob`.
        Args:
            circuits_ids (list of ints): circuits to display, if `None` (default) - prints all circuits.
        """
        if circuits_ids is None:
            circuits_ids = [i for i in range(1, len(self.circuits_ob))]
                
        for c_id in circuits_ids:
            print(f"Ansatz {c_id}:")
            display(self.circuits_ob[c_id].decompose().draw('mpl'))
    
    def run_vqe_circuits(self, circuits_ids, backend, process_iters=1, optimizer_iters=200, csv_file=None):
        """
        Runs VQE (with SPSA optimizer) for the given `circuit_ids` on `backend`, `process_iters` times.
        Writes the results to a CSV file in path `csv_file` (if `csv_file` is `None` a unique name is given to a new file, see implementation below).
        
        Args:
            circuits_ids(list of ints) - ids of the circuits to run VQE with.
            backend - backend to run the algorithm upon.
            process_iters(int) - iterations over the whole algorithm, per circuit.
            optimizer_iters(int) - iterations setting for the SPSA optimizer.
            csv_file(str) - a CSV file's path to write the results to.
            
        """

        optimizer = SPSA(maxiter=optimizer_iters)

        try:
            backend_name = backend.name()
        except TypeError:
            backend_name = backend.name

        if csv_file is None:
            csv_file = f"data/{time.time()}_VQE_NoiseExp_results_{self.name}_{backend_name}_{process_iters}_{optimizer_iters}.csv"

        with open(csv_file, 'w') as f_data:
            csv_writer = csv.writer(f_data)

            # Setting columns
            csv_writer.writerow(['c_id','iter_id','exact_eigenvalue','min_eigenvalue_approx','energy_diff'])

            # Running VQE for each circuit `process_iters` times
            for c_id in circuits_ids:
                for i in range(1, process_iters + 1):
                    print(f"Running VQE, ansatz circuit_id = {c_id}, iteration = {i}")
                    vqe_ob = VQE(ansatz = self.circuits_ob[c_id], optimizer=optimizer, quantum_instance=backend)
                    vqe_res = vqe_ob.compute_minimum_eigenvalue(self.H)

                    # Writing the record to the CSV file
                    approx_mean_ev = float(vqe_res.eigenvalue + self.NRE)
                    energy_diff = approx_mean_ev - float(self.exact_min_ev)
                    csv_writer.writerow([c_id, i, self.exact_min_ev, approx_mean_ev, energy_diff])
                        
        print(f"DONE. Results written into {csv_file}")

    def csv_data_to_table(self, csv_file, caption=None):
        """
        Transforms experiment data from a CSV file into a pandas.DataFrame object.
        The best record is chosen from multiple records for the same ansatz circuit (same `c_id`).

        Args:
            csv_file(str) - a path to a CSV file contains the data experiments for ansatzes in self.cicuits_ob (some or all).
            caption(str) - a caption for the presented table, default=`None` i.e no caption.

        Returns: {df, df_styler}
            df(DataFrame object) - the actual parsed data.
            df_styler(Styler object) - the Styler object to be presented.
        """

        csv_data = pd.read_csv(csv_file)

        best_records = []
        for i in range(1, len(self.circuits_ob)):
            sub_df = csv_data[csv_data['c_id'] == i]
            if len(sub_df) > 0:
                record = {'c_id': i,'min_eigenvalue_approx': sub_df['min_eigenvalue_approx'].min(),
                        'energy_diff': sub_df['energy_diff'].min()}
                best_records.append(record)

        df = pd.DataFrame(best_records)
        df.sort_values('energy_diff', inplace=True)
        df.rename(columns={'c_id': 'Circuit Id', 'min_eigenvalue_approx': 'GSE approximation', 'energy_diff': 'Energy difference'}, inplace=True)

        df_styler = df.style
        df_styler.format(precision=4)
        df_styler.hide(axis='index')
        df_styler.set_table_styles([{'selector': 'td', 'props': 'text-align: center;'}], overwrite=False)
        if caption is not None:
            df_styler.set_caption(caption)
            df_styler.set_table_styles([{'selector': 'caption', 'props': 'text-align: center; color: black; font-size: 120%; font-weight: bold;'}], overwrite=False)

        return {'df': df, 'df_styler': df_styler}

    def p_est(self, backends, csv_file=None, shots=1000):
        """
        Runs a heuristic method for estimating the probabilistic error rate of an ansatz w.r.t to a backend.
        The heuristic method is described in the main work paper, section 5 - https://github.com/ohadlev77/VQE_Noise/raw/main/paper.pdf.
        
        Args:
            backends - list of backends.
            csv_file(str) - a CSV file's path to write the results to.
            shots(int) - number of shot to use for each p-estimation (defualt=1000).
        """

        if csv_file is None:
            csv_file = f"data/{time.time()}_p_est.csv"

        zero_n = '0' * self.n
            
        with open(csv_file, 'w') as f:
            csv_writer = csv.writer(f)
            backends_names = list(map(lambda x: x.name, backends))
            columns = ['Circuit_ID'] + backends_names
            csv_writer.writerow(columns)
            
            for i, qc in enumerate(self.ent_parts_only):
                if qc is not None:
                    qc.measure_all()
                    
                    b_results = [i]
                    for b in backends:
                        tpqc = transpile(qc, b)
                        job = b.run(tpqc, shots=shots)
                        counts = job.result().get_counts()
                        p = (shots - counts[zero_n]) / shots
                        b_results.append(p)
                    
                    csv_writer.writerow(b_results)
                    print(f"Done with circuit {i}")
        
        print(f"DONE writing to {csv_file}")
                