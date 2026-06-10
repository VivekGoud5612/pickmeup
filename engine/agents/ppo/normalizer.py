import torch






class Value_Normalizer:
    def __init__(self, epsilon : float = 1e-5):

        self.epsilon = epsilon

        self.running_mean = 0.0
        self.running_var = 1.0

        self.count = epsilon   #Global count for step rewards. Initialized to epsilon,to prevent zero division on first update


    def update(self, returns : torch.Tensor) -> None:

        batch_mean = torch.mean(returns).item()
        batch_var = torch.var(returns).item()
        batch_count = returns.numel()

        total_count = self.count + batch_count

        # Difference between this batch's mean and old historical mean
        delta = batch_mean - self.running_mean

        # Shift the running mean towards the new batch mean
        self.running_mean += delta * (batch_count/total_count)

        # Welford's and Chad's Algorithm
        # For running variance we calculate the sum of differences for both old data(ma) and new batch(mb).
        # Then we combine them using a correction factor for how much the mean just shifted (delta**2)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count)/total_count

        # Convert M2 to running variance 
        self.running_var = M2 / total_count

        self.count = total_count


    def Normalize(self, returns : torch.Tensor) -> torch.Tensor:
        # Standard deviation is square root of Variance
        std = (self.running_var + self.epsilon) ** 0.5

        return (returns - self.running_mean) / std
    

    def Denormalize(self, values : torch.Tensor) -> torch.Tensor:

        std = (self.running_var + self.epsilon) ** 0.5

        return (values * std) + self.running_mean
    
