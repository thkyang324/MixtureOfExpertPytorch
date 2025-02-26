import torch
from torch import nn


class SimpleGate(nn.Module):
    def __init__(self, input_dims, num_experts, K, num_shared_experts=0, gate_encoder=None, *args, **kwargs):
        super(SimpleGate, self).__init__()

        assert K <= num_experts, "K must be less than num_experts"
        assert K + num_shared_experts <= num_experts, "K + num_shared_experts must be less than or equal to num_experts"

        if gate_encoder is None:
            self.gate_encoder = nn.Sequential(
                nn.Linear(input_dims, num_experts),
            )

        elif isinstance(gate_encoder, nn.Module):
            for module in gate_encoder:
                out_features = -1
                if isinstance(module, nn.Linear):
                    out_features = module.out_features
            assert out_features == num_experts, "The output features of the last layer must be equal to num_experts"
            self.gate_encoder = gate_encoder
            
        else:
            raise ValueError("gate_encoder must be a nn.Module")
        
        self.K = K
        self.num_shared_experts = num_shared_experts
    
    def forward(self, inputs):
        logits = self.gate_encoder(inputs)
        _shared, _specialized = logits[:, :self.num_shared_experts], logits[:, self.num_shared_experts:]
        _specialized_topk_logits, _specialized_topk_indices = torch.topk(_specialized, self.K, dim=-1)
        shared_indices = torch.arange(self.num_shared_experts).reshape(1, -1).expand(inputs.shape[0], -1).to(_specialized.device)
        topk_indices = torch.cat([shared_indices, _specialized_topk_indices + self.num_shared_experts], dim=1)
        topk_logits = torch.cat([_shared, _specialized_topk_logits], dim=1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        return topk_probs, topk_indices, logits


def simulate_gate(input_dims, num_experts, K, num_shared_experts, batch_size):
    gate = SimpleGate(input_dims, num_experts, K, num_shared_experts)
    inputs = torch.randn(batch_size, input_dims)
    topk_probs, topk_indices, logits = gate(inputs)

    assert topk_probs.shape == (batch_size, K + num_shared_experts), f"Expected topk_probs shape {(batch_size, K + num_shared_experts)}, but got {topk_probs.shape}"
    assert topk_indices.shape == (batch_size, K + num_shared_experts), f"Expected topk_indices shape {(batch_size, K + num_shared_experts)}, but got {topk_indices.shape}"
    assert logits.shape == (batch_size, num_experts), f"Expected logits shape {(batch_size, num_experts)}, but got {logits.shape}"
    print("simulate_gate passed.")


if __name__ == '__main__':
    for input_dims, num_experts, K, num_shared_experts, batch_size in [(10, 5, 2, 1, 3), (10, 5, 2, 1, 5), (10, 5, 2, 0, 5)]:
        simulate_gate(input_dims, num_experts, K, num_shared_experts, batch_size)
    print("All tests passed.")
