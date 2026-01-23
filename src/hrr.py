import torch
from src.config import HDC_DIM, SEED

def bind(t1, t2):
    """
    HDC Binding Operation (XOR).
    t1, t2: Tensor of {-1, 1}
    in binary space {0, 1}: xor(a, b)
    in bipolar space {-1, 1}: mul(a, b)
    """
    return torch.mul(t1, t2)

def bundle(tensor_list):
    """
    HDC Superposition Operation (Majority Rule).
    tensor_list: List of tensors or stacked tensor
    Returns: Sign(Sum(tensors))
    """
    if isinstance(tensor_list, list):
        stacked = torch.stack(tensor_list, dim=0)
    else:
        stacked = tensor_list
        
    s = torch.sum(stacked, dim=0)
    # Sign function: >0 -> 1, <0 -> -1, 0 -> 1 (bias to 1)
    res = torch.sign(s)
    res[res == 0] = 1.0
    return res

class HDCWorldModel:
    """
    Learns transition dynamics T(s, a) -> s' using HRR with Permutation.
    Trace = s * a * Permute(s')
    Pred = InversePermute(M * s * a)
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.dim = HDC_DIM
        # We store the sum as floats/ints to allow counting
        self.M_sum = torch.zeros(self.dim, device=self.device)
        self.count = 0
        
    def _permute(self, t, shifts=1):
        """Cyclic shift for permutation."""
        return torch.roll(t, shifts=shifts, dims=-1)
        
    def _inverse_permute(self, t, shifts=1):
        return torch.roll(t, shifts=-shifts, dims=-1)

    def learn(self, state, action, next_state):
        """
        One-shot learning of a transition.
        Trace = s * a * P(s_next)
        """
        # Ensure inputs are tensors on device
        s = state.to(self.device).float()
        a = action.to(self.device).float()
        ns = next_state.to(self.device).float()
        
        # Apply Permutation to Next State to break symmetry
        # This makes the bond directional: (s, a) -> ns
        p_ns = self._permute(ns)
        
        # Calculate Trace
        # trace = s * a * P(ns)
        trace = bind(bind(s, a), p_ns)
        
        # Accumulate
        self.M_sum += trace
        self.count += 1
        
    def predict(self, state, action):
        """
        Predict next state.
        Pred = P_inv( Sign(M_sum * s * a) )
        """
        s = state.to(self.device).float()
        a = action.to(self.device).float()
        
        # Use raw sum
        M = self.M_sum
        
        # Unbind: result = M * s * a
        # Expected result = P(ns) + Noise
        res = bind(bind(M, s), a)
        
        # Retrieve P(ns) via sign (denoise slightly)
        # Note: Binarizing here helps clean up before inverse permute
        res_binary = torch.sign(res)
        res_binary[res_binary == 0] = 1.0
        
        # Inverse Permute to get ns
        pred = self._inverse_permute(res_binary)
        
        return pred
