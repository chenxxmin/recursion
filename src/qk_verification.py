import sys
import os
import torch

# Allow running from repo root as: python src/qk_verification.py <pth> [init_len]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Shared implementations live in analyze_attention.py; this script is a thin CLI wrapper.
from analyze_attention import load_model, verify_qk_properties


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qk_verification.py <path_to_pth> [init_len]")
        print("Example: python qk_verification.py fibonacci_transformer.pth 2")
        sys.exit(1)

    pth_path = sys.argv[1]
    init_len = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, checkpoint = load_model(pth_path, device=device)
    config = checkpoint['config']
    p = config['p']

    import random
    max_len = config.get('block_size', 20)
    seq = [random.randint(0, p - 1) for _ in range(init_len)]
    for i in range(init_len, max_len):
        if 'a' in config and 'b' in config:
            a, b = config['a'], config['b']
            seq.append((a * seq[-1] + b * seq[-2]) % p)
        else:
            seq.append((seq[-1] + seq[-2]) % p)

    verify_qk_properties(model, [seq], init_len=init_len, device=device)
