import torch
from torch import nn

class TinyNeRF_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_encoding_level = 6
        self.direction_encoding_level = 4
        position_encoding_feature_length = 3 + 3 * 2 * self.position_encoding_level
        direction_encoding_feature_length = 3 + 3 * 2 * self.direction_encoding_level

        network_width = 256
        self.early_mlp = nn.Sequential(
            nn.Linear(position_encoding_feature_length, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width + 1),
            nn.ReLU(),
        )
        self.late_mlp = nn.Sequential(
            nn.Linear(network_width + direction_encoding_feature_length, network_width),
            nn.ReLU(),
            nn.Linear(network_width, 3),
            nn.Sigmoid(),
        )

    def forward(self, positions, directions):
        positions_encoded = [positions]
        for l_pos in range(self.position_encoding_level):
            positions_encoded.append(torch.sin(2**l_pos * torch.pi * positions))
            positions_encoded.append(torch.cos(2**l_pos * torch.pi * positions))

        positions_encoded = torch.cat(positions_encoded, dim=-1)

        directions = directions / directions.norm(p=2, dim=-1).unsqueeze(-1)
        directions_encoded = [directions]
        for l_dir in range(self.direction_encoding_level):
            directions_encoded.append(torch.sin(2**l_dir * torch.pi * directions))
            directions_encoded.append(torch.cos(2**l_dir * torch.pi * directions))

        directions_encoded = torch.cat(directions_encoded, dim=-1)

        outputs = self.early_mlp(positions_encoded)
        sigma_is = outputs[:, 0]
        c_is = self.late_mlp(torch.cat([outputs[:, 1:], directions_encoded], dim=-1))
        return {"c_is": c_is, "sigma_is": sigma_is}