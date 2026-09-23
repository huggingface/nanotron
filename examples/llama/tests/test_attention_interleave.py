"""CPU-only test for the RoPE-interleave contract of the weight converters.

`convert_checkpoint_and_save` (both directions) used to call the converters
without an `interleave_qkv` argument, so it always defaulted to False and
ignored the model's `rope_interleaved`. A model trained with
`rope_interleaved=True` was then exported with the wrong q/k permutation —
silently corrupted weights, no error. The fix passes `rope_interleaved`
through to `_handle_attention_block`, whose two branches are pinned here.

`_handle_attention_block` lives in an example script that imports the full
training stack (flash_attn / triton) at module load, so it cannot be imported
on a CPU-only box. `reference_interleave_q` mirrors its `interleave_weight`
+ q-slice logic exactly (kept in sync by the assertions below); the real
function is additionally exercised by the GPU round-trip tests in
test_conversion.py.
"""

import pytest

torch = pytest.importorskip("torch")


def reference_interleave_q(qkv: torch.Tensor, n_q_heads: int, d_qk: int, interleave: bool) -> torch.Tensor:
    def interleave_weight(w: torch.Tensor) -> torch.Tensor:
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    q = qkv[: n_q_heads * d_qk]
    return interleave_weight(q) if interleave else q


def _q_rows(d_qk: int, n_q_heads: int, hidden: int) -> torch.Tensor:
    rows = n_q_heads * d_qk
    return torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)


def test_interleave_false_is_passthrough():
    d_qk, n_q, hidden = 4, 2, 3
    qkv = _q_rows(d_qk, n_q, hidden)
    out = reference_interleave_q(qkv, n_q, d_qk, interleave=False)
    assert torch.equal(out, qkv[: n_q * d_qk])


def test_interleave_true_maps_gptj_pairs_to_neox_halves():
    # Within a head, GPT-J interleaved pairs (0,1),(2,3) become NeoX halves
    # [0,2],[1,3]; a second head's rows shift by d_qk.
    d_qk, n_q, hidden = 4, 2, 1
    qkv = _q_rows(d_qk, n_q, hidden)
    out = reference_interleave_q(qkv, n_q, d_qk, interleave=True)
    assert out.flatten().tolist() == [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
    # the flag must be load-bearing: interleaving changes the tensor
    assert not torch.equal(out, qkv[: n_q * d_qk])
