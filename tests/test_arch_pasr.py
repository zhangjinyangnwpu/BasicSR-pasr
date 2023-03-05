import sys
sys.path.append("..")

import torch
from pasr.archs.pasr_arch import PASR

def test_pasr_arch():
    """Test arch: PASR."""
    device = 'cuda' if torch.cuda.is_available() else 'mps' # or cpu
    print(f"use {device} testing")
    # model init and forward
    net = PASR(input_channels=3, output_channels=3, scale=4, num_layers=5, fea_dim=32).to(device)
    img = torch.rand((1, 3, 56, 56), dtype=torch.float32).to(device)
    output = net(img)
    assert output.shape == (1, 3, 224, 224)
    print("PASR x4 test finished")

if __name__ == '__main__':
    test_pasr_arch()