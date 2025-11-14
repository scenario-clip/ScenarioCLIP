from contextlib import contextmanager
import random, numpy as np, torch

@contextmanager
def temp_seed(seed: int, touch_cuda: bool = False):
    py_state  = random.getstate()
    np_state  = np.random.get_state()
    th_state  = torch.get_rng_state()
    cu_states = torch.cuda.get_rng_state_all() if (touch_cuda and torch.cuda.is_available()) else None
    try:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if cu_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(th_state)
        if cu_states is not None:
            torch.cuda.set_rng_state_all(cu_states)

