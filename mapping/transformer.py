import torch
from torch import nn
import einops


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        inner_dim = dim_head * n_head
        project_out = not (n_head == 1 and dim_head == input_dim)

        self.n_head = n_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(input_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, input_dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layer: int,
        dim_head: int,
        n_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadAttention(input_dim, n_head, dim_head, dropout),
                        MLPLayer(input_dim, hidden_dim, dropout),
                    ]
                )
            )

        self.out_layer = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x

        x = self.out_layer(x)
        return self.norm(x)
