"""Regression coverage for restoring the scalar RNG after a resident packet."""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from rec_mlx.kernels.masked_sampling import FilteredResidentSampler


def test_masked_packet_restores_seeded_scalar_state():
    sampler = FilteredResidentSampler.__new__(FilteredResidentSampler)
    sampler.engine = SimpleNamespace(model=SimpleNamespace(eos_tokens=[1]))
    sample = make_sampler(temp=0.8, top_p=0.95, top_k=2)
    logits = mx.array([0.0, 0.0, -float("inf")])

    def run(*args):
        tokens, keys = [], []
        for _ in range(args[6]):
            tokens.append(sample(logits - mx.logsumexp(logits)))
            keys.append(mx.random.state[0])
        # Simulate EOS after the first token: the rest of the packet is discarded.
        return (mx.stack(tokens), mx.array([True, False, False]), mx.zeros((3,)),
                mx.zeros((3,)), mx.zeros((3,)), mx.stack(keys))

    sampler.run = run

    def packet(seed):
        mx.random.seed(seed)
        state = {"metadata": None, "kv": SimpleNamespace(banks=[]), "position": 0,
                 "initial": 0, "capacity": 3}
        tokens = sampler.block(state, 0, 3, temperature=0.8, top_p=0.95, top_k=2)
        following = int(sample(logits - mx.logsumexp(logits)).item())
        return tokens, following, state["position"]

    for seed in range(8):
        mx.random.seed(seed)
        expected = [int(sample(logits - mx.logsumexp(logits)).item()) for _ in range(2)]
        actual = packet(seed)
        assert actual == (expected[:1], expected[1], 1)
        assert packet(seed) == actual
        assert all(token != 2 for token in (*actual[0], actual[1]))
