import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

class CTRNN:
    def __init__(self, size=0):
        self.size = size
        self.states = np.zeros(size)
        self.outputs = np.zeros(size)
        self.biases = np.zeros(size)
        self.gains = np.ones(size)
        self.taus = np.ones(size)
        self.Rtaus = np.ones(size)
        self.externalinputs = np.zeros(size)
        self.weights = np.zeros((size, size))

    def load_from_file(self, filepath):
        with open(filepath, 'r') as f:
            tokens = f.read().split()
            
        if not tokens:
            return False
            
        try:
            self.size = int(tokens[0])
            idx = 1
            self.__init__(self.size)
            
            for i in range(self.size):
                self.taus[i] = float(tokens[idx])
                self.Rtaus[i] = 1.0 / self.taus[i]
                idx += 1
                
            for i in range(self.size):
                self.biases[i] = float(tokens[idx])
                idx += 1
                
            for i in range(self.size):
                self.gains[i] = float(tokens[idx])
                idx += 1
                
            for i in range(self.size):
                for j in range(self.size):
                    self.weights[i][j] = float(tokens[idx])
                    idx += 1
            return True
        except Exception as e:
            print(f"Error parsing CTRNN: {e}")
            return False

    def euler_step(self, stepsize):
        inputs = self.externalinputs + np.dot(self.outputs, self.weights)
        self.states += stepsize * self.Rtaus * (inputs - self.states)
        self.outputs = sigmoid(self.gains * (self.states + self.biases))
