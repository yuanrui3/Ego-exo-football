"""Single-boundary AMP accounting, shared by detector training and diagnostics.

The caller accumulates scaled backwards before calling update(). Schedulers remain
caller-owned: Ultralytics advances its schedule by epoch, including skipped updates.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


class AmpSkipLimit(RuntimeError):
    pass


@dataclass
class UpdateCounters:
    minibatches: int = 0
    attempts: int = 0
    committed: int = 0
    amp_skipped: int = 0
    consecutive_skips: int = 0


class AmpGuard:
    def __init__(self, *, max_consecutive: int = 5, max_window_skips: int = 10,
                 window_size: int = 100, event_sink: Callable[[dict], None] | None = None):
        if min(max_consecutive, window_size) < 1 or max_window_skips < 0:
            raise ValueError("Invalid configurable skip limits")
        self.counters = UpdateCounters()
        self.max_consecutive = max_consecutive
        self.max_window_skips = max_window_skips
        self.window_size = window_size
        self.window: deque[bool] = deque(maxlen=window_size)
        self.event_sink = event_sink
        self.first_abnormal: dict | None = None

    def state_dict(self) -> dict:
        return {"counters": asdict(self.counters), "window": list(self.window),
                "max_consecutive": self.max_consecutive,
                "max_window_skips": self.max_window_skips, "window_size": self.window_size,
                "first_abnormal": self.first_abnormal}

    def load_state_dict(self, state: dict) -> None:
        if (state["max_consecutive"], state["max_window_skips"], state["window_size"]) != (
                self.max_consecutive, self.max_window_skips, self.window_size):
            raise ValueError("Restored skip limits differ")
        self.counters = UpdateCounters(**state["counters"])
        self.window = deque(state["window"], maxlen=self.window_size)
        self.first_abnormal = state["first_abnormal"]

    def note_minibatch(self, loss: torch.Tensor, context: dict | None = None, *,
                       scaler: torch.amp.GradScaler | None = None) -> None:
        self.counters.minibatches += 1
        if not bool(torch.isfinite(loss.detach()).all()):
            event = {"status": "NONFINITE_LOSS", "loss": str(loss.detach().cpu().tolist()),
                     "context": context or {}, "counters": asdict(self.counters),
                     "precision": "AMP_FP16" if scaler is not None and scaler.is_enabled() else "FP32",
                     "scale_before": float(scaler.get_scale()) if scaler is not None else 1.,
                     "scale_after": float(scaler.get_scale()) if scaler is not None else 1.,
                     "affected_parameters": []}
            self._emit(event, abnormal=True)
            raise RuntimeError("Nonfinite loss")

    def _emit(self, event: dict, *, abnormal: bool) -> None:
        if abnormal and self.first_abnormal is None:
            self.first_abnormal = event.copy()
            event["first_abnormal"] = True
        if self.event_sink is not None:
            self.event_sink(event)

    def update(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler, loss: torch.Tensor, *,
               ema: Any = None, context: dict | None = None) -> dict:
        self.counters.attempts += 1
        scale_before = float(scaler.get_scale())
        event = {"status": "FAILED", "loss": str(loss.detach().cpu().tolist()),
                 "scale_before": scale_before, "precision": "AMP_FP16" if scaler.is_enabled() else "FP32",
                 "context": context or {}, "affected_parameters": []}
        try:
            if not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError("Nonfinite loss")
            named = list(model.named_parameters())
            bad = [name for name, p in named if not bool(torch.isfinite(p.detach()).all())]
            if bad:
                event["affected_parameters"] = bad
                raise RuntimeError("Nonfinite model parameters")
            gradients = [(name, p) for name, p in named if p.grad is not None]
            missing = [name for name, p in named if p.requires_grad and p.grad is None]
            if not gradients or missing:
                event["affected_parameters"] = missing
                raise RuntimeError("Missing gradients")
            scaler.unscale_(optimizer)
            bad = [name for name, p in gradients if not bool(torch.isfinite(p.grad).all())]
            event["affected_parameters"] = bad
            if bad and not scaler.is_enabled():
                raise RuntimeError("Nonfinite FP32 gradients")
            if not bad:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite clipping norm")
                event["unscaled_gradient_norm"] = float(norm)
            # GradScaler owns overflow detection and scale backoff; never clip NaN/Inf.
            scaler.step(optimizer)
            scaler.update()
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize(next(model.parameters()).device)
            if bad:
                self.counters.amp_skipped += 1
                self.counters.consecutive_skips += 1
                event["status"] = "AMP_NONFINITE_SKIPPED"
            else:
                self.counters.committed += 1
                self.counters.consecutive_skips = 0
                event["status"] = "COMMITTED"
                bad_parameters = [name for name, p in named if not bool(torch.isfinite(p.detach()).all())]
                if bad_parameters:
                    event["affected_parameters"] = bad_parameters
                    raise RuntimeError("Nonfinite model parameters after optimizer update")
                if ema is not None:
                    ema.update(model)
            self.window.append(bool(bad))
            event.update(scale_after=float(scaler.get_scale()), counters=asdict(self.counters))
            if bad and event["scale_after"] >= scale_before:
                raise RuntimeError("AMP overflow did not reduce the loss scale")
            stop = (self.counters.consecutive_skips >= self.max_consecutive or
                    (len(self.window) == self.window_size and sum(self.window) > self.max_window_skips))
            event["stop_reason"] = "AMP_SKIP_LIMIT" if stop else None
            self._emit(event, abnormal=bool(bad))
            if stop:
                raise AmpSkipLimit("AMP_SKIP_LIMIT")
            return event
        except AmpSkipLimit:
            raise
        except Exception as exc:
            event.update(status="FAILED", error=str(exc), counters=asdict(self.counters),
                         scale_after=float(scaler.get_scale()))
            self._emit(event, abnormal=True)
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)
