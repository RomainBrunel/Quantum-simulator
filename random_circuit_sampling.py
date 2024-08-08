from n_mode_simulator import *
import itertools as it

gate_set = ["H","S","X","Y","Z", "Cnot", "Rx", "Ry", "Rz"] # List of the game implemented in the simulator

rng = np.random.default_rng() # Define a random instance
depth = 10 # Define the depth of the circuit
dimension = 5 # Define the number of qubitthe system is made of

rdm_gate = rng.choice(gate_set, size=depth)

Cnot_arg = list(it.permutations(range(dimension),2))
gate_arg = range(dimension)

# Create an array to store the arguments
rdm_arg = [[tuple(rng.choice(Cnot_arg))] if gate == "Cnot" else [rng.choice(gate_arg)] for gate in rdm_gate]


rdm_angles = rng.random(rdm_gate.shape) * 2 * np.pi
rdm_angles = np.array(rdm_angles[((rdm_gate == "Rx") | (rdm_gate == "Ry") | (rdm_gate == "Rz"))])
rdm_angles = [[i] for i in rdm_angles]
print(rdm_angles)
sim = quantum_simulator(dimension=dimension)

sim.add_Rz_gate(1,[1])
sim.add_Rx_gate(1,[1])
sim.add_H_gate(1)
print(sim.circuit)
print(sim.theta)
sim.clear_circuit()
sim.circuit = rdm_gate
sim.circuit_args = rdm_arg
sim.theta = rdm_angles
print(sim.circuit)
print(sim.theta)
vector = np.zeros(2**dimension)
vector[0]= 1
state = quantum_state(dimension, vector=vector)

sim.run(state)
state.plot_state(True)