import torch
from dynamicemb.dynamicemb_config import DynamicEmbInitializerArgs, DynamicEmbInitializerMode
from dynamicemb.initializer import DebugInitializer
def test_debug_initializer_fills_key_mod_100000() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    dtype = torch.float32
    # Debug initializer ignores args fields other than mode.
    args = DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.DEBUG)
    init = DebugInitializer(args)
    num_rows = 128
    dim = 16
    buffer = torch.zeros((num_rows, dim), device=device, dtype=dtype)
    # keys must be indexable by row-id in `buffer` (vec_id in CUDA kernel).
    keys = torch.arange(num_rows, device=device, dtype=torch.int64) * 12345 + 7
    indices = torch.tensor([0, 1, 2, 5, 63, 127], device=device, dtype=torch.int64)
    init(buffer, indices, keys)
    out = buffer[indices].detach()
    expected = (keys[indices] % 100000).to(dtype).view(-1, 1).expand(-1, dim)
    assert torch.allclose(out, expected)
    # Sanity: rows not in `indices` should remain zero.
    mask = torch.ones(num_rows, device=device, dtype=torch.bool)
    mask[indices] = False
    assert torch.all(buffer[mask] == 0)
