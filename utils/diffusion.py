import torch

class Diffusion:
    def __init__(self, device, diffusion_steps=1000, image_size=64, beta_1=1e-4, beta_T=0.02):
        # Initialize class variables
        self.diffusion_steps = diffusion_steps
        self.device = device
        self.beta_1 = beta_1
        self.beta_T = beta_T

        # Compute beta decay schedule
        self.compute_beta_schedule()

    def compute_beta_schedule(self):
        self.beta = torch.linspace(self.beta_1, self.beta_T, self.diffusion_steps, device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def apply_noise(self, x_0):
        """
        Apply noise to batch of image tensors based on beta decay schedule at a uniformly distributed time step sample.
        """
        # Sample timestep: t ∼ Uniform({1, . . . , T})
        t = torch.randint(1, self.diffusion_steps + 1, (x_0.shape[0],), device=self.device)

        # Sample noise: epsilon ~ N(0,I)
        epsilon = torch.randn_like(x_0, device=self.device)

        # Apply noise to image based on beta schedule
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t-1]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t-1]).view(-1, 1, 1, 1)
        x_t = (sqrt_alpha_bar_t * x_0) + (sqrt_one_minus_alpha_bar_t * epsilon)

        return x_t, t, epsilon
    
    def remove_noise(self, x_t, t, epsilon_pred):
        if t > 1:
            z = torch.normal(0, 1, x_t.size(), device=self.device)
        else:
            z = torch.zeros(x_t.size(), device=self.device)

        inv_sqrt_alpha_t = 1 / torch.sqrt(self.alpha[t-1])
        one_minus_alpha_t = 1 - self.alpha[t-1]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t-1])
        subtract = (one_minus_alpha_t / sqrt_one_minus_alpha_bar_t) * epsilon_pred
        sigma_t = torch.sqrt(self.beta[t-1])

        return inv_sqrt_alpha_t * (x_t - subtract) + sigma_t * z
        

    def state_dict(self):
        """
        Get the current state of the diffusion model for checkpointing.
        """
        return {
            'diffusion_steps': self.diffusion_steps,
            'beta_1': self.beta_1,
            'beta_T': self.beta_T,
            'alpha_t': self.alpha,
            'alpha_bar_t': self.alpha_bar
        }

    def load_state_dict(self, state):
        """
        Load the state of the diffusion model from a checkpoint.
        """
        self.diffusion_steps = state['diffusion_steps']
        self.beta_1 = state['beta_1']
        self.beta_T = state['beta_T']
        self.compute_beta_schedule()