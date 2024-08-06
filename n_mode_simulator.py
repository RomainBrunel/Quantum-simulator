import numpy as np
import matplotlib.pyplot as plt
import itertools as it

class quantum_state():
    def __init__(self, dimension: int = 2) -> None:
        """ Class that define the quantum state that will be used for the computation
        arg:
            - dimension : int, number of qubit in the quantum state
        methods:
            - .input_state: ndarray, correpsonding to the stae initial state vefore computation
            - .output_state: ndarrau, cooressponding to the stae after the computation
        
        In this class the states are defined in the normal bases {0,1}^dimension
        For dimension equal 2 : {|00>, |01>, |10>, |11>}"""
        
        self.basis = list(it.product([0, 1], repeat=dimension))
        self.dimension = dimension
        self.input_state = None
        self.output_state = None
        self.generate_rdm_state()
    
        
    def generate_rdm_state(self):
        random_state = np.random.rand(2**self.dimension)
        normalized_state = random_state / np.linalg.norm(random_state)
        self.input_state = normalized_state

    def plot_state(self):
        label = ["|"+"".join(map(str,i))+">"for i in self.basis]
        plt.figure(figsize=(10, 6))
        plt.bar(range(2**self.dimension), self.input_state, color='blue', tick_label = label)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.ylim(-0.1,1)
        plt.title('N-Dimensional State Vector Bar Chart')
        plt.grid(True)
        plt.show()

class one_qubit_gate():

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
    
    def I(self):
        return np.eye(2)
    
    def zero(self):
        return np.array([[1,0],[0,0]])
    
    def one(self):
        return np.array([[0,0],[0,1]])
    
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
    
class quantum_simulator():
    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension  
        self.one_qgate = one_qubit_gate()

    def H(self, modes:list):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.H())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def S(self, modes:list, theta):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.S(theta))
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def X(self, modes:list):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Y(self, modes:list):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Z(self, modes:list):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Cnot(self, modes):
        zero = 1
        one = 1
        c = np.array(modes)[:,0]
        q = np.array(modes)[:,1]
        print(c,q)
        for i in range(self.dimension):
            if i in c:
                zero = np.kron(zero,self.one_qgate.zero())
                one = np.kron(one,self.one_qgate.one())
            elif i in q:
                zero = np.kron(zero,self.one_qgate.I())
                one = np.kron(one,self.one_qgate.X()) 
            else:
                zero = np.kron(zero,self.one_qgate.I())
                one = np.kron(one,self.one_qgate.I())
        unitary = zero + one
        return unitary

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



