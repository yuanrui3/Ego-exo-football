import copy

import pytest
import torch

from src.backbone.amp_guard import AmpGuard, AmpSkipLimit


class EMA:
    updates = 0

    def update(self, model):
        self.updates += 1


def setup(device="cpu", *, fused=False, **limits):
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, fused=fused)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda", init_scale=128.)
    return model, optimizer, scaler, AmpGuard(**limits), EMA()


def backward(model, scaler, guard):
    loss = model(torch.ones(2, 2, device=next(model.parameters()).device)).square().mean()
    guard.note_minibatch(loss)
    scaler.scale(loss).backward()
    return loss


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_finite_update_changes_parameters_and_accounting():
    m, o, s, g, e = setup()
    before = copy.deepcopy(m.state_dict())
    loss = backward(m, s, g)
    event = g.update(m, o, s, loss, ema=e)
    assert event["status"] == "COMMITTED"
    assert g.counters.committed == g.counters.attempts == g.counters.minibatches == 1
    assert e.updates == 1
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
@pytest.mark.parametrize("fused", [False, True])
def test_real_cuda_overflow_skip_then_finite_and_restore(tmp_path, fused):
    m, o, s, g, e = setup("cuda", fused=fused)
    g.update(m, o, s, backward(m, s, g), ema=e)
    loss = backward(m, s, g)
    next(m.parameters()).grad.fill_(float("inf"))
    params, state = copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict())
    scale, ema_updates = s.get_scale(), e.updates
    event = g.update(m, o, s, loss, ema=e)
    assert event["status"] == "AMP_NONFINITE_SKIPPED"
    assert_state_equal(params, m.state_dict())
    assert_state_equal(state, o.state_dict())
    assert s.get_scale() < scale
    assert e.updates == ema_updates
    assert g.counters.committed == 1 and g.counters.amp_skipped == 1
    assert all(p.grad is None for p in m.parameters())
    checkpoint = tmp_path / "disposable.pt"
    torch.save({"guard": g.state_dict(), "scaler": s.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    restored = AmpGuard()
    restored.load_state_dict(saved["guard"])
    restored_scaler = torch.amp.GradScaler("cuda")
    restored_scaler.load_state_dict(saved["scaler"])
    assert restored.state_dict() == g.state_dict()
    assert restored_scaler.state_dict() == s.state_dict()
    g.update(m, o, s, backward(m, s, g), ema=e)
    assert g.counters.committed == 2 and g.counters.attempts == 3
    assert g.counters.consecutive_skips == 0
    assert e.updates == 2


@pytest.mark.parametrize("failure", ["loss", "gradient", "missing", "parameter", "norm"])
def test_hard_failures(failure):
    m, o, s, g, e = setup()
    loss = backward(m, s, g)
    if failure == "loss":
        loss = loss * float("inf")
    elif failure == "gradient":
        next(m.parameters()).grad.fill_(float("inf"))
    elif failure == "missing":
        next(m.parameters()).grad = None
    elif failure == "parameter":
        next(m.parameters()).data.fill_(float("inf"))
    else:
        for p in m.parameters():
            p.grad.fill_(1e30)
    with pytest.raises(RuntimeError):
        g.update(m, o, s, loss, ema=e)
    assert g.counters.committed == 0
    assert e.updates == 0
    assert g.first_abnormal is not None
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
@pytest.mark.parametrize("window_limit", [False, True])
def test_configurable_skip_limits(window_limit):
    limits = ({"max_consecutive": 100, "max_window_skips": 1, "window_size": 4}
              if window_limit else {"max_consecutive": 2})
    m, o, s, g, e = setup("cuda", **limits)
    for i in range(4 if window_limit else 2):
        loss = backward(m, s, g)
        if not window_limit or i % 2 == 0:
            next(m.parameters()).grad.fill_(float("inf"))
        if i == (3 if window_limit else 1):
            with pytest.raises(AmpSkipLimit):
                g.update(m, o, s, loss, ema=e)
        else:
            g.update(m, o, s, loss, ema=e)
    assert g.counters.amp_skipped == 2
    assert g.counters.attempts == g.counters.committed + g.counters.amp_skipped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
def test_accumulation_unscales_once_only_at_boundary(monkeypatch):
    m, o, s, g, e = setup("cuda")
    calls = []
    original = s.unscale_
    def unscale(optimizer):
        calls.append(True)
        return original(optimizer)
    monkeypatch.setattr(s, "unscale_", unscale)
    backward(m, s, g)
    assert not calls and g.counters.attempts == 0
    loss = backward(m, s, g)
    g.update(m, o, s, loss, ema=e)
    assert len(calls) == 1
    assert g.counters.minibatches == 2 and g.counters.committed == 1


def test_nonfinite_minibatch_records_complete_metadata():
    m, o, s, g, e = setup()
    context = {"fitting_samples": ["fitting.jpg"], "rng": "available local state",
               "augmentation": "native training configuration"}
    with pytest.raises(RuntimeError, match="Nonfinite loss"):
        g.note_minibatch(torch.tensor(float("nan")), context, scaler=s)
    assert g.first_abnormal["precision"] == "FP32"
    assert g.first_abnormal["scale_before"] == g.first_abnormal["scale_after"] == 1.
    assert g.first_abnormal["context"] == context
    assert g.counters.attempts == 0
