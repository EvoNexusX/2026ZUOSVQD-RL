"""Count interactions at env.step, with a carried offset for resumed runs.
Evaluation calls, evaluation episodes and gradient updates are separately
reported counters; they include only explicitly instrumented reporting sites."""
from typing import Any, Dict, Optional
import weakref

__all__ = ["EnvStepAccountant", "attach_counter", "accountant_snapshot"]


class EnvStepAccountant:
    """Counts real `env.step` calls plus the three auxiliary cost counters."""

    def __init__(self) -> None:
        self.env_steps: int = 0
        self.eval_episodes: int = 0
        self.eval_calls: int = 0
        self.grad_updates: int = 0
        self.base_offset: int = 0
        self._wrapped: "weakref.WeakSet" = weakref.WeakSet()
        self._by_env: Dict[int, str] = {}

    @property
    def total_env_steps(self) -> int:
        """Cumulative env steps including anything inherited from a resume."""
        return int(self.base_offset) + int(self.env_steps)

    def wrap(self, env, source: str = "unknown") -> Any:
        """Install a counting proxy on `env.step` (idempotent per env object)."""
        if env is None:
            return env
        try:
            if env in self._wrapped:
                return env
        except TypeError:
            if getattr(env, '_qd_step_accountant', None) is self:
                return env

        raw_step = env.step
        counter = self
        label = str(source)

        def counted_step(*args, **kwargs):
            out = raw_step(*args, **kwargs)
            counter.env_steps += 1
            return out

        try:
            env.step = counted_step
        except Exception:
            return env

        try:
            self._wrapped.add(env)
            self._by_env[id(env)] = label
        except TypeError:
            pass
        try:
            env._qd_step_accountant = self
        except Exception:
            pass
        return env

    def add_eval_episodes(self, n: int) -> None:
        try:
            self.eval_episodes += max(0, int(n))
        except Exception:
            pass

    def add_eval_call(self, n: int = 1) -> None:
        try:
            self.eval_calls += max(0, int(n))
        except Exception:
            pass

    def add_grad_updates(self, n: int) -> None:
        try:
            self.grad_updates += max(0, int(n))
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            'env_steps': int(self.env_steps),
            'total_env_steps': int(self.total_env_steps),
            'base_offset': int(self.base_offset),
            'eval_episodes': int(self.eval_episodes),
            'eval_calls': int(self.eval_calls),
            'grad_updates': int(self.grad_updates),
            'envs_wrapped': len(self._wrapped),
        }


def attach_counter(env, accountant: Optional[EnvStepAccountant], source: str = "unknown"):
    """Attach `accountant` to `env` if there is one; return `env` either way."""
    if accountant is None or env is None:
        return env
    return accountant.wrap(env, source=source)


def accountant_snapshot(accountant: Optional[EnvStepAccountant]) -> Dict[str, Any]:
    if accountant is None:
        return {
            'env_steps': 0, 'total_env_steps': 0, 'base_offset': 0,
            'eval_episodes': 0, 'eval_calls': 0, 'grad_updates': 0,
            'envs_wrapped': 0,
        }
    return accountant.snapshot()
