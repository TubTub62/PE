

class filter_layers:

    def __init__(self, num_layers, f_size, k_size, activation):

        self.num_layers = num_layers
        self.f_size = f_size
        self.k_size = k_size
        self.activation = activation

class dense_layers:

    def __init__(self, num_layers, d_size, activation):

        self.num_layers = num_layers
        self.d_size = d_size
        self.activation = activation

class auxilliary:

    def __init__(self, dropout, learning_rate):
        
        self.dropout = dropout
        self.learning_rate = learning_rate