import numpy as np
import matplotlib.pyplot as plt
import itertools as it

class quantum_state():
    def __init__(self, dimension: int = 2, vector:None|np.ndarray = None) -> None:
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
        if vector is None : 
            self.generate_rdm_state()
        else:
            self.generate_state(vector)
    
    def generate_state(self, vector):
        if len(vector) == 2**self.dimension:
            self.input_state = vector / np.linalg.norm(vector)
        else:
            print(f"Wrong dimension for the vector, must be dimension {2**self.dimension}")
        
    def generate_rdm_state(self):
        random_state = np.random.rand(2**self.dimension)
        normalized_state = random_state / np.linalg.norm(random_state)
        self.input_state = normalized_state

    def plot_state(self, in_out: bool = False):
        """ True for output state
            False for input state"""
        if in_out:
            s = self.output_state
        else:
            s = self.input_state
        label = ["|"+"".join(map(str,i))+">"for i in self.basis]
        plt.figure(figsize=(10, 6))
        plt.bar(range(2**self.dimension), np.abs(s)**2, color='blue', tick_label = label)
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
    
    def S(self, *args):
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
    
    def Rx(self,theta):
        return np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]])
    
    def Ry(self,theta):
        return np.array([[np.cos(theta/2),-1*np.sin(theta/2)],[1*np.sin(theta/2),np.cos(theta/2)]])
    
    def Rz(self,theta):
        return np.array([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]])
    
class quantum_simulator():
    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension  
        self.one_qgate = one_qubit_gate()
        self.circuit = []
        self.circuit_args = []
        self.theta =[]

    def H(self, modes:list, *args):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.H())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def S(self, modes:list, theta, *args):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.S(theta))
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def X(self, modes:list, *args):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Y(self, modes:list, *args):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Z(self, modes:list, *args):
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.X())
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Rx(self, modes:list, theta:list, *args):
        """The Modes must be place in increasing orderand angles must much the modes
        arg:
            -modes : list of modes to apply the gates
            -theta : list of angles to apply to the modes"""
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.Rx(theta[i]))
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Ry(self, modes:list, theta, *args):
        """The Modes must be place in increasing orderand angles must much the modes
        arg:
            -modes : list of modes to apply the gates
            -theta : list of angles to apply to the modes"""
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.Ry(theta[i]))
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Rz(self, modes:list, theta, *args):
        """The Modes must be place in increasing orderand angles must much the modes
        arg:
            -modes : list of modes to apply the gates
            -theta : list of angles to apply to the modes"""
        unitary = 1
        for i in range(self.dimension):
            if i in modes:
                unitary = np.kron(unitary,self.one_qgate.Rz(theta[i]))
            else:
                unitary = np.kron(unitary,self.one_qgate.I()) 
        return unitary
    
    def Cnot(self, modes, *args):
        """
        arg:
            - modes: list of tuple where a tuple represent the modes you want to apply the gate (control_mode, target_mode)
                    can looks like [(0,1),(2,8)]"""
        zero = 1
        one = 1
        c = np.array(modes)[:,0]
        q = np.array(modes)[:,1]
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

    def generate_circuit(self, gates:dict, *args):
        """ Function that generate the unitary matrix corresponding to the matrix of the quantum circuit
        arg: 
            gates: dict name of all the gate apply in the circuit. First element is aply first, then second, and so on so forth
                   value is used for the parameter of the gate 
        return:
            unitary: unitary matrix of the quantum simulator"""
        unitary = np.eye(2)
        gate_set = {"H":self.H,"S":self.S,"X":self.X,"Y":self.Y,"Z":self.Z, "Cnot":self.Cnot}
        for gate in gates:
            if gate in gate_set:
                unitary = unitary @ gate_set[gate](gates[gate])
            else:


                print(f"{gate} is not from the game set: {gate_set}")
        return unitary
    
    def add_H_gate(self, *args):
        self.circuit.append("H")
        self.circuit_args.append(args)
    
    def add_X_gate(self, *args):
        self.circuit.append("X")
        self.circuit_args.append(args)

    def add_Y_gate(self, *args):
        self.circuit.append("Y")
        self.circuit_args.append(args)
    
    def add_Z_gate(self, *args):
        self.circuit.append("Z")
        self.circuit_args.append(args)

    def add_S_gate(self, *args):
        self.circuit.append("S")
        self.circuit_args.append(args)

    def add_Rx_gate(self, theta, *args):
        """theta must be a list of angles to apply to the modes
        MUST BE ORDERED IN INCREASING ORDER"""
        self.circuit.append("Rx")
        self.theta.append(theta)
        self.circuit_args.append((theta,args))

    def add_Ry_gate(self, theta,*args):
        """theta must be a list of angles to apply to the modes
        MUST BE ORDERED IN INCREASING ORDER"""
        self.circuit.append("Ry")
        self.theta.append(theta)
        self.circuit_args.append((theta,args))

    def add_Rz_gate(self, theta,*args):
        """theta must be a list of angles to apply to the modes
        MUST BE ORDERED IN INCREASING ORDER"""
        self.circuit.append("Rz")
        self.theta.append(theta)
        self.circuit_args.append(args)

    def add_Cnot_gate(self, *args):
        """
        args: list of tuple where a tuple represent the modes you want to apply the gate (control_mode, target_mode)
                    can looks like [(0,1),(2,8)]"""
        self.circuit.append("Cnot")
        self.circuit_args.append(*args)

    def run(self, state:quantum_state):
        gate_set = {"H":self.H,"S":self.S,"X":self.X,"Y":self.Y,"Z":self.Z, "Cnot":self.Cnot, "Rx": self.Rx, "Ry": self.Ry, "Rz": self.Rz}
        state.output_state = state.input_state.copy()
        i=0
        for gate, arg in zip(self.circuit, self.circuit_args):
            if gate in ["Rx","Ry","Rz"]:
                state.output_state = gate_set[gate](arg,self.theta[i]) @ state.output_state
                i+=1
            else:
                state.output_state = gate_set[gate](arg) @ state.output_state
        return state.output_state

    def clear_circuit(self):
        self.circuit = []
        self.circuit_args = []
        self.theta =[]
    
        

if __name__ == "__main__":
    state = quantum_state(3, np.array([1,0,0,0,0,0,0,0]))
    print(state.input_state)
    sim = quantum_simulator(3)
    # a = sim.H([0]) @ sim.Cnot([(0,2)]) @ np.array([1,0,0,0,0,0,0,0])
    sim.add_Rz_gate([np.pi/4],0)
    sim.add_Cnot_gate([(0,1)])
    print(sim.circuit_args)
    sim.run(state)
    state.plot_state(in_out = True)
    state.plot_state(in_out = False)