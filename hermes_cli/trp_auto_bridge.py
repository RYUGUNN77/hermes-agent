"""Bootstrap the virtual TRP_AUTO provider onto a physical subscription runtime."""

from __future__ import annotations

from contextlib import suppress
from contextvars import ContextVar
import importlib.util
import logging
from types import ModuleType
from typing import Any

from hermes_constants import get_hermes_home


_RESOLVING: ContextVar[bool] = ContextVar("trp_auto_bootstrap_resolving", default=False)
_FORBIDDEN_PHYSICAL_PROVIDERS = frozenset({"", "auto", "custom", "moa", "openrouter", "trp-auto"})
logger = logging.getLogger(__name__)


class TrpAutoBootstrapError(RuntimeError):
    """TRP_AUTO could not resolve a safe physical bootstrap seat."""


def _load_seat_module() -> ModuleType:
    path = get_hermes_home() / "plugins" / "trp-auto" / "seat.py"
    if not path.is_file():
        raise TrpAutoBootstrapError(f"TRP_AUTO plugin is missing seat.py under {path.parent}.")
    spec = importlib.util.spec_from_file_location("_hermes_trp_auto_seat", path)
    if spec is None or spec.loader is None:
        raise TrpAutoBootstrapError("TRP_AUTO seat.py could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TrpAutoBootstrapError(
            f"TRP_AUTO seat.py failed to load ({type(exc).__name__})."
        ) from None
    return module


def _decision_field(decision: Any, name: str) -> str:
    value = getattr(decision, name, None)
    if not isinstance(value, str) or not value.strip():
        raise TrpAutoBootstrapError(f"TRP_AUTO bootstrap decision has invalid {name}.")
    return value.strip()


def _resolve_physical_runtime(provider: str, model: str) -> dict:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested=provider, target_model=model)


def resolve_bootstrap_runtime(target_model: str | None = None) -> dict:
    """Resolve ``TRP_AUTO`` to its plugin-selected physical subscription runtime."""
    if target_model is not None and str(target_model).strip().casefold() != "trp_auto".casefold():
        raise TrpAutoBootstrapError("TRP_AUTO accepts only the TRP_AUTO virtual model.")
    if _RESOLVING.get():
        raise TrpAutoBootstrapError("TRP_AUTO virtual-provider recursion was blocked.")

    token = _RESOLVING.set(True)
    try:
        module = _load_seat_module()
        bootstrap = getattr(module, "bootstrap_seat", None)
        if not callable(bootstrap):
            raise TrpAutoBootstrapError("TRP_AUTO seat.py does not define bootstrap_seat().")
        try:
            decision = bootstrap()
        except TrpAutoBootstrapError:
            raise
        except Exception as exc:
            raise TrpAutoBootstrapError(
                f"TRP_AUTO bootstrap_seat() failed ({type(exc).__name__})."
            ) from None

        provider = _decision_field(decision, "provider").lower()
        model = _decision_field(decision, "model")
        reason = _decision_field(decision, "reason")
        control_digest = _decision_field(decision, "control_digest")
        frontier = getattr(decision, "frontier", None)
        switch = getattr(decision, "switch", None)
        if not isinstance(frontier, bool) or switch is not True:
            raise TrpAutoBootstrapError("TRP_AUTO bootstrap decision has invalid flags.")
        if provider in _FORBIDDEN_PHYSICAL_PROVIDERS:
            raise TrpAutoBootstrapError(
                f"TRP_AUTO bootstrap decision selected non-physical provider '{provider or '<empty>'}'."
            )
        try:
            runtime = _resolve_physical_runtime(provider, model)
        except Exception as exc:
            raise TrpAutoBootstrapError(
                f"TRP_AUTO could not resolve subscription seat {provider}/{model} ({type(exc).__name__})."
            ) from None
        if not isinstance(runtime, dict):
            raise TrpAutoBootstrapError("TRP_AUTO physical resolver returned an invalid runtime.")
        physical = str(runtime.get("provider") or "").strip().lower()
        if physical in _FORBIDDEN_PHYSICAL_PROVIDERS:
            raise TrpAutoBootstrapError(
                f"TRP_AUTO physical resolver returned non-physical provider '{physical or '<empty>'}'."
            )
        if physical != provider:
            raise TrpAutoBootstrapError(
                f"TRP_AUTO physical runtime '{physical}' did not match bootstrap provider '{provider}'."
            )

        result = dict(runtime)
        result["requested_provider"] = "trp-auto"
        result["trp_auto"] = {
            "virtual_provider": "trp-auto",
            "virtual_model": "TRP_AUTO",
            "bootstrap_provider": provider,
            "bootstrap_model": model,
            "reason": reason,
            "frontier": frontier,
            "control_digest": control_digest,
        }
        return result
    finally:
        _RESOLVING.reset(token)


def _warn_turn_route(agent: Any, error: Exception) -> None:
    """Report a fail-closed turn route without exposing exception text or credentials."""
    message = f"TRP_AUTO kept the current subscription seat ({type(error).__name__})."
    logger.warning(message)
    emit = getattr(agent, "_emit_warning", None)
    if callable(emit):
        with suppress(Exception):
            emit(message)


def _turn_uses_trp_auto(agent: Any) -> bool:
    current = str(getattr(agent, "requested_provider", "") or "").strip().lower()
    primary = getattr(agent, "_primary_runtime", None)
    primary_requested = (
        str(primary.get("requested_provider") or "").strip().lower()
        if isinstance(primary, dict)
        else ""
    )
    return "trp-auto" in {current, primary_requested}


def route_turn(agent: Any, user_text: str) -> bool:
    """Route one TRP_AUTO turn to a physical subscription seat in place."""
    if not _turn_uses_trp_auto(agent):
        return False

    try:
        current_provider = str(getattr(agent, "provider", "") or "").strip().lower()
        current_model = str(getattr(agent, "model", "") or "").strip()
        if current_provider in _FORBIDDEN_PHYSICAL_PROVIDERS or not current_model:
            raise TrpAutoBootstrapError("TRP_AUTO current runtime is not a physical seat.")

        module = _load_seat_module()
        resolve = getattr(module, "resolve_seat", None)
        if not callable(resolve):
            raise TrpAutoBootstrapError("TRP_AUTO seat.py does not define resolve_seat().")
        decision = resolve(
            user_text,
            requested_provider="trp-auto",
            current_provider=current_provider,
            current_model=current_model,
        )
        provider = _decision_field(decision, "provider").lower()
        model = _decision_field(decision, "model")
        _decision_field(decision, "reason")
        _decision_field(decision, "control_digest")
        frontier = getattr(decision, "frontier", None)
        switch = getattr(decision, "switch", None)
        if not isinstance(frontier, bool) or not isinstance(switch, bool):
            raise TrpAutoBootstrapError("TRP_AUTO turn decision has invalid flags.")
        if provider in _FORBIDDEN_PHYSICAL_PROVIDERS:
            raise TrpAutoBootstrapError("TRP_AUTO turn decision selected a non-physical provider.")
        if not switch:
            return False

        runtime = _resolve_physical_runtime(provider, model)
        if not isinstance(runtime, dict):
            raise TrpAutoBootstrapError("TRP_AUTO physical resolver returned an invalid runtime.")
        physical = str(runtime.get("provider") or "").strip().lower()
        if physical != provider or physical in _FORBIDDEN_PHYSICAL_PROVIDERS:
            raise TrpAutoBootstrapError("TRP_AUTO physical runtime did not match the turn decision.")

        agent.switch_model(
            new_model=model,
            new_provider=provider,
            api_key=runtime.get("api_key") or "",
            base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "",
            capabilities=runtime.get("capabilities"),
        )
        agent.requested_provider = "trp-auto"
        primary = getattr(agent, "_primary_runtime", None)
        if not isinstance(primary, dict):
            raise TrpAutoBootstrapError("TRP_AUTO model switch did not publish a primary runtime.")
        primary["requested_provider"] = "trp-auto"
        return True
    except Exception as exc:
        _warn_turn_route(agent, exc)
        return False
