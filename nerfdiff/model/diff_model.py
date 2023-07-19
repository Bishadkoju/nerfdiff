import torch
from torch import nn
from nerfdiff.model.positional_emb import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class DiffModel(nn.Module):
    def __init__(self, 
                 hidden_size: int = 1024, 
                 hidden_layers: int = 5, 
                 emb_size: int = 128,
                 time_emb: str = "sinusoidal", 
                 input_emb: str = "sinusoidal"
                ):
        super().__init__()

        #! Embeddings for image
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
        self.label_mlp = nn.Linear(1000, 512)

        #! Positional Embeddings for time
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        #! Positional Embeddings for 7 dim input
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp3 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp4 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp5 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp6 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp7 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
                      512 + \
                      len(self.input_mlp1.layer) + \
                      len(self.input_mlp2.layer) + \
                      len(self.input_mlp3.layer) + \
                      len(self.input_mlp4.layer) + \
                      len(self.input_mlp5.layer) + \
                      len(self.input_mlp6.layer) + \
                      len(self.input_mlp7.layer)


        #! Input Layer
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]

        #! Hidden Layers
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))

        #! Output Layer
        layers.append(nn.Linear(hidden_size, 7)) #TODO Dont jump directly to 7 here !!

        #! Wrap all the layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, l):

        #! Input Embeddings
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        x3_emb = self.input_mlp2(x[:, 2])
        x4_emb = self.input_mlp2(x[:, 3])
        x5_emb = self.input_mlp2(x[:, 4])
        x6_emb = self.input_mlp2(x[:, 5])
        x7_emb = self.input_mlp2(x[:, 6])

        #! Time Embedding
        t_emb = self.time_mlp(t)

        #! Image Embedding
        l_emb = self.resnet(l)
        l_emb = self.label_mlp(l_emb)

        x = torch.cat((x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, x7_emb, t_emb, l_emb), dim=-1)
        x = self.joint_mlp(x)

        return x