import torch
import torch.nn.functional as F
from src.config import MHN_BETA, HDC_DIM

class ModernHopfieldNetwork:
    """
    Modern Hopfield Network (Dense Associative Memory).
    Uses Softmax attention mechanism for retrieval.
    Task 2.1: The Mind (Memory)
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.beta = MHN_BETA
        # Memory storage: List of tensors or a single stacked tensor
        # We start with empty.
        self.memory_matrix = torch.empty(0, HDC_DIM, device=self.device)
        
    def add_memory(self, pattern):
        """
        Add a new pattern to memory.
        pattern: (HDC_DIM,) tensor or numpy array
        """
        if not isinstance(pattern, torch.Tensor):
            pattern = torch.from_numpy(pattern).float().to(self.device)
        else:
            pattern = pattern.float().to(self.device)
            
        # Ensure shape (1, D)
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        # Append to memory matrix
        self.memory_matrix = torch.cat([self.memory_matrix, pattern], dim=0)
        
    def retrieve(self, query):
        """
        Retrieve the closest memory pattern.
        Formula: X_new = Softmax(beta * X * M^T) * M
        Args:
            query: (HDC_DIM,) or (Batch, HDC_DIM)
        Returns:
            recalled: Binary restored pattern
        """
        if self.memory_matrix.shape[0] == 0:
            return query # No memory, return query as is
            
        if not isinstance(query, torch.Tensor):
            query = torch.from_numpy(query).float().to(self.device)
        else:
            query = query.float().to(self.device)
            
        # Handle batch
        is_batch = query.dim() > 1
        if not is_batch:
            query = query.unsqueeze(0) # (1, D)
            
        # 1. Similarity (Energy)
        # Query: (B, D), Memory: (N, D)
        # E = Q * M^T -> (B, N)
        similarity = torch.mm(query, self.memory_matrix.T)
        
        # 2. Attention (Softmax)
        # Weighting over memories
        weights = F.softmax(self.beta * similarity, dim=-1) # (B, N)
        
        # 3. Reconstruction
        # Recon = W * M -> (B, N) * (N, D) -> (B, D)
        reconstruction = torch.mm(weights, self.memory_matrix)
        
        # 4. Binarize (Sign)
        # Auto-associative task usually requires clean output
        output = torch.sign(reconstruction)
        output[output == 0] = 1.0
        
        if not is_batch:
            return output.squeeze(0)
            
        return output
    
    @property
    def memory_count(self):
        return self.memory_matrix.shape[0]

    def compute_energy(self, query):
        """
        Compute the scalar energy of the query state against memories.
        E = -lse(beta * X * M^T)
        """
        if self.memory_count == 0: return 0.0
        
        if not isinstance(query, torch.Tensor):
            query = torch.from_numpy(query).float().to(self.device)
        else:
            query = query.float().to(self.device)
            
        if query.dim() == 1: query = query.unsqueeze(0)
        
        similarity = torch.mm(query, self.memory_matrix.T) # (1, N)
        energy = -torch.logsumexp(self.beta * similarity, dim=-1)
        
        return energy.item()
