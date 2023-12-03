import math

class NeuronWarmup(object):
    def __init__(self, eta0, Tmax, ) -> None:
        self.eta0 = eta0
        self.Tmax = Tmax
        self.alpha = eta0
        self.last_epoch = 0
    
    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        
    def get_last_alpha(self):
        return self.alpha
    
    def step(self):
        self._update_alpha()
        
        if self.last_epoch != 0 and self.last_epoch % self.Tmax == 0:
            self.alpha = self.eta0
        
        self.last_epoch += 1
    
    def _update_alpha(self):
        raise NotImplementedError('')
    
    def __call__(self):
        return self.alpha

# step
class StepWarmup(NeuronWarmup):
    def __init__(self, eta0, Tmax, step_size, gamma) -> None:
        super().__init__(eta0, Tmax)
        self.step_size = step_size
        self.gamma = gamma
    
    def _update_alpha(self):
        if (1 + self.last_epoch) % self.step_size == 0:
            self.alpha *= self.gamma

# exponential
class ExponentialWarmup(StepWarmup):
    def __init__(self, eta0, Tmax, gamma) -> None:
        super().__init__(eta0, Tmax, step_size=1, gamma=gamma)

# polynormial
class PolynormialWarmup(NeuronWarmup):
    def __init__(self, eta0, eta_max, Tmax) -> None:
        super().__init__(eta0, Tmax)
        self.eta_max = eta_max
    
    def _update_alpha(self):
        self.alpha += (self.eta_max - self.eta0) / float(self.Tmax)
        
# cosine annealing
class ConsineAnnealingWarmup(NeuronWarmup):
    def __init__(self, eta0, eta_max, Tmax) -> None:
        super().__init__(eta0, Tmax)
        self.eta_max = eta_max
        
    def _update_alpha(self):
        self.alpha = self.eta0 + (self.eta_max - self.eta0) * (1 - math.cos(2 * math.pi * float(1 + self.last_epoch) / float(self.Tmax))) / 2
        