import numpy as np
import matplotlib.pyplot as plt

class quantum_state():
    def __init__(self) -> None:
        self.input_state = None
        self.output_state = None
        
    def generate_rdm_state(self):
        random_state = np.random.randn(2)
        normalized_state = random_state / np.linalg.norm(random_state)
        self.state = normalized_state
    
    def generate_state(self, c0, c1):
        state = np.array(c0,c1)/np.sqrt(c0**2+c1**2)
        self.state = state

    def plot_state(self, in_out:bool = False):
        """ True for output state
            False for input state"""
        if in_out:
            s = self.output_state
        else:
            s = self.input_state

        plt.figure("Quantum state")
        fig, ax = plt.subplots()
        ax.quiver(0, 0, s[0], s[1], angles='xy', scale_units='xy', scale=1)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.grid(True)
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D State Vector Plot')
        plt.show()     

class one_qubit_simulator():
    def __init__(self, gates:dict) -> None:
        """arg: 
            gates: dict name of all the gate apply in the circuit. First element is aply first, then second, and so on so forth
                   value is used for the parameter of the gate
                   """
        self.input_state = None
        self.circuit = None
        self.circuit = self.generate_circuit(gates)

    def H(self, *args):
        return 1/np.sqrt(2)*np.array([[1,1],
                                      [1,-1]])
    
    def S(self, theta, *args):
        return np.array([[1,0],
                         [0,1j]])
    
    def X(self, *args):
        return np.array([[0,1],
                         [1,0]])
    
    def Y(self, *args):
        return np.array([[0,-1j],
                         [1j,0]])
    
    def Z(self, *args):
        return np.array([[1,0],
                         [0,-1]])
    def R(self, *args):
        M = np.random.rand(2,2)
        return M/np.linalg.norm(M)
    
    def generate_circuit(self, gates:dict):
        """ Function that generate the unitary matrix corresponding to the matrix of the quantum circuit
        arg: 
            gates: dict name of all the gate apply in the circuit. First element is aply first, then second, and so on so forth
                   value is used for the parameter of the gate 
        return:
            unitary: unitary matrix of the quantum simulator"""
        unitary = np.eye(2)
        gate_set = {"H":self.H,"S":self.S,"X":self.X,"Y":self.Y,"Z":self.Z, "R":self.R}
        for gate in gates:
            if gate in gate_set:
                unitary = unitary @ gate_set[gate](gates[gate])
            else:
                print(f"{gate} is not from the game set: {gate_set}")
        return unitary
    
    def run_simulation(self, input_state: quantum_state):
        output_state = self.circuit @ input_state.state
        input_state.output_state = output_state
        return output_state

