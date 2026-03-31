import torch

from dynamicemb.dynamicemb_config import DynamicEmbInitializerArgs, DynamicEmbInitializerMode
from dynamicemb.initializer import TruncatedNormalInitializer


def test_truncated_normal_initializer_bounds_and_sanity() -> None:
    if not torch.cuda.is_available():
        # This initializer is implemented in CUDA extension kernels.
        return

    device = torch.device("cuda")
    dtype = torch.float32

    mean = 0.0
    std_dev = 1.0
    low_bound = mean - 2 * std_dev
    high_bound = mean + 2 * std_dev

    args = DynamicEmbInitializerArgs(
        mode=DynamicEmbInitializerMode.TRUNCATED_NORMAL,
        mean=mean,
        std_dev=std_dev
    )
    init = TruncatedNormalInitializer(args)

    num_rows = 8192
    dim = 128
    buffer = torch.zeros((num_rows, dim), device=device, dtype=dtype)

    idx = torch.tensor([0, 7, 11, 1024, 4096, 8191], device=device, dtype=torch.int64)
    # keys is unused by TruncatedNormalInitializer, but the interface requires it.
    keys = idx

    init(buffer, idx, keys)

    init_tensor = buffer[idx].detach()

    assert init_tensor.shape == (idx.numel(), dim)

    init_min = init_tensor.min().item()
    init_max = init_tensor.max().item()
    assert init_min >= low_bound - 1e-6, f"min:{init_min} lower than expected {low_bound}"
    assert init_max <= high_bound + 1e-6, f"max:{init_max} higher than expected {high_bound}"
    assert not torch.all(init_tensor == 0), (
        "All initialized values from TruncatedNormalInitializer are 0!"
    )

    # Basic sanity: should not degenerate to a constant vector.
    s = init_tensor.float().std(unbiased=False).item()
    assert s > 0.01

