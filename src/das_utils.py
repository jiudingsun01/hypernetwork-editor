import torch
from pyvene import TrainableIntervention, DistributedRepresentationIntervention
from pyvene.models.intervenable_base import LowRankRotateLayer
from pyvene.models.intervention_utils import _can_use_fast

# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/layers.py#L16 
class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid(
        (boundary_y - _input) / temperature
    )

def get_rotation_mask(subspace_proj):
    """
    """
    intervention_boundaries = torch.clamp(subspace_proj.intervention_boundaries, 1e-3, 1)
    boundary_mask = sigmoid_boundary(
        subspace_proj.intervention_population.repeat(1, 1),
        0.0,
        intervention_boundaries[0] * int(subspace_proj.embed_dim),
        subspace_proj.temperature
    )
    return boundary_mask

def compute_rotation_mask_sparsity(subspace_proj):
    """
    """
    rotation_mask = get_rotation_mask(subspace_proj)
    return (rotation_mask.sum() / rotation_mask.numel()).item()

# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py#L298
class BoundlessRotatedSpaceIntervention(torch.nn.Module):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries
    
    def get_boundary_sparsity(self):
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(1, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        
        return boundary_mask.sum() / boundary_mask.numel()
        
        

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=True
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )

        boundary_mask = (
            torch.ones_like(base)[:,0].to(base.device) * boundary_mask
            # torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        ).unsqueeze(dim=1).expand(base.shape)
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention()"
    
    

class RotatedSpaceIntervention(torch.nn.Module):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, intervention_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.intervention_dim = intervention_dim
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        intervention_mask = torch.zeros(self.embed_dim)
        intervention_mask[: self.intervention_dim] = 1
        self.intervention_mask = torch.nn.Parameter(
            intervention_mask.unsqueeze(0), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_mask
    
    def get_boundary_sparsity(self):
        return self.intervention_mask.sum() / self.intervention_mask.numel()
    
    def get_temperature(self):
        pass

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=False
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        boundary_mask = self.intervention_mask.repeat(batch_size, 1)

        boundary_mask = (
            torch.ones_like(base)[:,0].to(base.device) * boundary_mask
        ).unsqueeze(dim=1).expand(base.shape)
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"RotatedSpaceIntervention()"
    
    
class LowRankRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=False)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.sparsity = kwargs["low_rank_dimension"] / self.embed_dim
        
    def get_boundary_parameters(self):
        return None
    
    def get_boundary_sparsity(self):
        return self.sparsity
    
    def get_temperature(self):
        pass

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        pass

    def forward(self, base, source, batch_size=None):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        
        output = base + torch.matmul(
            (rotated_source - rotated_base), self.rotate_layer.weight.T
        )
        
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention()"