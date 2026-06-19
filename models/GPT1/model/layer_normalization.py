import torch
import torch.nn as nn


class LayerNormalization(nn.Module):

    def __init__(self, input_shape):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(input_shape))
        self.shift = nn.Parameter(torch.zeros(input_shape))

        self.epsilon = 1e-5


    def forward(self, x):
        dims = tuple(range(-len(self.scale.shape), 0))

        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, unbiased=False, dim=dims, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.epsilon)

        x = x * self.scale + self.shift

        return x


if __name__ == "__main__":
    import time

    input_shape = (512, 128)
    input_data = torch.randn((32, 512, 128))

    ln1 = LayerNormalization(input_shape)
    ln2 = nn.LayerNorm(input_shape)

    input_data = input_data.to('mps')
    ln1 = ln1.to('mps')
    ln2 = ln2.to('mps')

    s = time.time()
    for _ in range(100):
        out1 = ln1.forward(input_data)
    print((time.time() - s) * 1000)

    s = time.time()
    for _ in range(100):
        out2 = ln2(input_data)
    print((time.time() - s) * 1000)

    # print(out1)
    # print(out1.shape)
    # print()
    # print(out2)
    # print(out2.shape)


