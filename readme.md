### How to generate a circuit
- Create an intance of the class quantum_simulator
- Add gate to your circuit using the function add_X_gate:
    - the argument of the one mode function is the qubits on witch those gate will be applied on.
    - the argument for the 2 mode gate Cnot is a list of tuple on which qubit you would like to apply the gates on. The first element of the tuple is the control qubit qnd the second is the target.
    - for rotation matrixes you need to pass as a list the angles of the rotation and then the qubits you want to apply them.
- Create an instance of quantum_state, you can initialize the state or let it be random.
- Run the simulation by inserting your quantum state as an argument of the run function.
Exemple of simulation: 
```
state = quantum_state(3, np.array([1,0,0,0,0,0,0,0]))
sim = quantum_simulator(3)
sim.add_Ry_gate([np.pi/2,np.pi],0,1)
sim.add_Cnot_gate([(1,2)])
print(sim.circuit_args)
sim.run(state)
state.plot_state(in_out = True)
state.plot_state(in_out = False)
```
