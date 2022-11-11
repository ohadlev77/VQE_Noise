"""
Contains the construction of qubit Hamiltonian operators objects (PauliSumOp objects).
It is required that PyQuante and Qiskit-Nature to be installed to run this module.
If PyQuante / Qiskit-Nature are not installed - please comment / delete SECTION 1 and uncomment SECTION 2.
"""

# SECTION 1 START

from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.settings import settings
settings.dict_aux_operators = True

# H_2 molecule to qubit Hamiltonian
d = 0.735 # Bond length in Angstroms
molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, d]]])
driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE)
es_problem = ElectronicStructureProblem(driver)
second_q_op = es_problem.second_q_ops()
qubit_converter = QubitConverter(mapper=JordanWignerMapper())

H_2_hamiltonian = qubit_converter.convert(second_q_op['ElectronicEnergy'], num_particles = es_problem.num_particles)
H_2_NRE = es_problem.grouped_property_transformed.get_property('ElectronicEnergy').nuclear_repulsion_energy

# SECTION 1 END
# ==============
# SECTION 2 START

# from qiskit.opflow import I, X, Y, Z

# H_2_hamiltonian =  (-0.8105479965981812 * (I ^ I ^ I ^ I)
#                     + 0.1721839435195097 * (I ^ I ^ I ^ Z)
#                     - 0.22575350027942803 * (I ^ I ^ Z ^ I)
#                     + 0.17218394351950977 * (I ^ Z ^ I ^ I)
#                     - 0.22575350027942806 * (Z ^ I ^ I ^ I)
#                     + 0.120912633445247 * (I ^ I ^ Z ^ Z)
#                     + 0.1689275406336891 * (I ^ Z ^ I ^ Z)
#                     + 0.04523280037093535 * (Y ^ Y ^ Y ^ Y)
#                     + 0.04523280037093535 * (X ^ X ^ Y ^ Y)
#                     + 0.04523280037093535 * (Y ^ Y ^ X ^ X)
#                     + 0.04523280037093535 * (X ^ X ^ X ^ X)
#                     + 0.16614543381618235 * (Z ^ I ^ I ^ Z)
#                     + 0.16614543381618235 * (I ^ Z ^ Z ^ I)
#                     + 0.17464343496147006 * (Z ^ I ^ Z ^ I)
#                     + 0.120912633445247 * (Z ^ Z ^ I ^ I))

# H_2_NRE = 0.7199690462585033

# SECTION 2 END