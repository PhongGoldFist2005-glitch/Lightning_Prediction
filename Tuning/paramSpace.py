import random

class hyper_parameters:
    def __init__(self, threshold, batch_size, alpha, gamma, lr, weight_decay, hidden_size, num_layer, drop_out):
        self.threshold = threshold
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.drop_out = drop_out
        self.lr = lr
        self.param_space = {
            "threshold": self.threshold,
            "batch_size":self.batch_size,
            "alpha":self.alpha,
            "gamma":self.gamma,
            "weight_decay":self.weight_decay,
            "hidden_size":self.hidden_size,
            "num_layer":self.num_layer,
            "drop_out":self.drop_out,
            "lr":self.lr
        }
    def random_sample(self):
        return {k: random.choice(v) for k, v in self.param_space.items()}