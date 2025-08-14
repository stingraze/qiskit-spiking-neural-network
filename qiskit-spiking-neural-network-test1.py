import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RYGate
from qiskit.primitives import StatevectorSampler


class QuantumSpikingNeuron:
    def __init__(self, num_inputs=3, threshold=0.5):
        self.num_inputs = num_inputs
        self.threshold = threshold
        self.weights = ParameterVector('w', num_inputs)
        self.input_params = ParameterVector('x', num_inputs)
        self.theta_param = Parameter('theta')
        self.bias_param = Parameter('bias')
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        """Build neuron with weighted input-controlled rotations"""
        qreg = QuantumRegister(self.num_inputs + 1, 'q')
        creg = ClassicalRegister(1, 'spike')
        qc = QuantumCircuit(qreg, creg)

        # Encode inputs as Ry rotations (binary input => 0 or pi/2)
        for i in range(self.num_inputs):
            qc.ry(self.input_params[i] * np.pi / 2, qreg[i])

        # Small bias rotation on output neuron
        qc.ry(self.bias_param, qreg[self.num_inputs])

        # Weighted controlled-RY from each input to output neuron
        for i in range(self.num_inputs):
            qc.append(RYGate(self.weights[i]).control(1), [qreg[i], qreg[self.num_inputs]])

        # Extra tunable rotation
        qc.ry(self.theta_param, qreg[self.num_inputs])

        # Measure output neuron
        qc.measure(qreg[self.num_inputs], creg[0])
        return qc

    def process_spike_train(self, spike_train, weights, theta=np.pi / 6, bias=np.pi / 12, shots=1024):
        sampler = StatevectorSampler()
        output_spikes, probabilities = [], []
        
        for spikes in spike_train:
            # Bind parameters
            params = {self.theta_param: theta, self.bias_param: bias}
            for i in range(self.num_inputs):
                params[self.input_params[i]] = float(spikes[i])
                params[self.weights[i]] = float(weights[i])

            bound = self.circuit.assign_parameters(params)

            # Run with shots to get BitArray with counts
            job = sampler.run([bound], shots=shots)
            pub_res = job.result()[0]  # SamplerPubResult
            
            # FIXED: BitArray doesn't have get_probabilities(), use get_counts()
            counts = pub_res.data.spike.get_counts()  # Returns dict like {"0": 512, "1": 512}
            
            # Calculate probability from counts
            total_shots = sum(counts.values())
            p1 = counts.get("1", 0) / total_shots if total_shots > 0 else 0.0
            
            probabilities.append(p1)
            output_spikes.append(1 if p1 > self.threshold else 0)

        return output_spikes, probabilities


class QuantumSpikingNetwork:
    def __init__(self, input_size=3, hidden_size=2, output_size=1):
        self.hidden_layer = [QuantumSpikingNeuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [QuantumSpikingNeuron(hidden_size) for _ in range(output_size)]
        self.hidden_weights = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.output_weights = np.random.uniform(-1, 1, (output_size, hidden_size))

    def forward(self, spike_train):
        hidden_spikes = []
        for i, neuron in enumerate(self.hidden_layer):
            spikes, _ = neuron.process_spike_train(spike_train, self.hidden_weights[i])
            hidden_spikes.append(spikes)

        hidden_spike_train = list(zip(*hidden_spikes))
        output_spikes = []
        for i, neuron in enumerate(self.output_layer):
            spikes, _ = neuron.process_spike_train(hidden_spike_train, self.output_weights[i])
            output_spikes.append(spikes)

        return output_spikes, hidden_spikes


# ---- DEMOS ----
def visualize_circuit():
    print("\n🎨 Example Quantum Spiking Neuron Circuit")
    print("=" * 50)
    neuron = QuantumSpikingNeuron()
    print(neuron.circuit.draw(output='text'))
    print(f"\nCircuit depth: {neuron.circuit.depth()}")
    print(f"Number of qubits: {neuron.circuit.num_qubits}")
    print(f"Number of parameters: {neuron.circuit.num_parameters}")


def demonstrate_spiking_patterns():
    print("\n🔬 Spiking Pattern Analysis")
    print("=" * 50)
    neuron = QuantumSpikingNeuron(threshold=0.3)
    patterns = [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ]
    weight_configs = [
        [0.8, 0.6, 0.4],
        [-0.5, -0.3, -0.7],
        [0.9, -0.5, 0.2],
    ]
    
    descriptions = ["All active", "Single spike", "Pattern A", "Pattern B", "Silent"]
    
    for idx, weights in enumerate(weight_configs):
        print(f"\nWeight Config {idx+1}: {weights}")
        print("Pattern | Spike Prob | Output | Description")
        print("-" * 48)
        
        spikes, probs = neuron.process_spike_train(patterns, weights)
        
        for pattern, p1, spike, desc in zip(patterns, probs, spikes, descriptions):
            print(f"{pattern} | {p1:9.3f} | {spike:6d} | {desc}")


def demonstrate_network():
    print("\n🧠 Network Behavior Demo")
    print("=" * 50)
    net = QuantumSpikingNetwork(input_size=3, hidden_size=2, output_size=1)
    patterns = {
        "Pattern A": [[1, 0, 1], [1, 0, 1]],
        "Pattern B": [[0, 1, 0], [0, 1, 0]],
        "Pattern C": [[1, 1, 0], [1, 1, 0]],
        "Random": [[random.choice([0, 1]) for _ in range(3)] for _ in range(2)]
    }
    
    print("Testing different input patterns:")
    print("\nPattern   | Hidden Activity | Output Activity")
    print("-" * 50)
    
    for name, data in patterns.items():
        out, hid = net.forward(data)
        hidden_rates = [sum(s)/len(s) for s in hid]
        output_rates = [sum(s)/len(s) for s in out]
        print(f"{name:9s} | {hidden_rates} | {output_rates}")


def classification_demo():
    print("\n📊 Simple Pattern Classification")
    print("=" * 50)
    net = QuantumSpikingNetwork(input_size=2, hidden_size=2, output_size=1)
    cases = [
        ([1, 0], "Type A"),
        ([0, 1], "Type A"),
        ([1, 1], "Type B"),
        ([0, 0], "Type B")
    ]
    
    print("Input | Expected | Network Output | Match")
    print("-" * 40)
    
    for inp, label in cases:
        data = [inp, inp]
        out_spikes, _ = net.forward(data)
        act = sum(out_spikes[0]) / len(out_spikes[0])
        pred = "Type A" if act > 0.5 else "Type B"
        match = "✓" if pred == label else "✗"
        print(f"{inp} | {label:8s} | {pred:13s} | {match}")


def run_comprehensive_demo():
    print("🚀 Quantum Spiking Neural Network - Complete Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    visualize_circuit()
    demonstrate_spiking_patterns()
    demonstrate_network()
    classification_demo()


if __name__ == "__main__":
    run_comprehensive_demo()
