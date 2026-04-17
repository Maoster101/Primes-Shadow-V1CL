"""Model capability profiles — auto-detected from Ollama's /api/show.

Instead of hardcoding what each model can do, we query Ollama for the
model's template and metadata, then derive capabilities from that.
The template is the source of truth — if it contains `think` blocks,
the model supports thinking. If it has `tool` references, it supports
tool calling.

Usage:
    from . import model_profiles
    profile = model_profiles.active()
    if profile.supports_think:
        ...
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Capability profile for a single Ollama model."""
    name: str                          # Ollama model tag
    family: str = "unknown"            # Model family (gemma3, gptoss, llama, etc.)
    parameter_size: str = "?"          # e.g. "12.2B"
    quantization: str = "?"            # e.g. "Q4_K_M"
    context_length: int = 8192         # Max context from model_info
    block_count: int = 0               # Transformer layers
    supports_think: bool = False       # Template has think/reasoning blocks
    supports_tools: bool = False       # Template has tool/function blocks
    supports_vision: bool = False      # Has vision encoder blocks
    notes: str = ""


# Active profile — set when model loads/switches
_active: ModelProfile = ModelProfile(name="none")


async def detect(model_name: str) -> ModelProfile:
    """Query Ollama /api/show and derive a capability profile.

    This is the ONLY place we need model-specific knowledge.
    Everything is derived from what Ollama tells us about the model.
    """
    import httpx

    profile = ModelProfile(name=model_name)

    try:
        async with httpx.AsyncClient(
            base_url="http://localhost:11434", timeout=10.0
        ) as client:
            resp = await client.post("/api/show", json={"name": model_name})
            resp.raise_for_status()
            data = resp.json()

        # Details
        details = data.get("details", {})
        profile.family = details.get("family", "unknown")
        profile.parameter_size = details.get("parameter_size", "?")
        profile.quantization = details.get("quantization_level", "?")

        # Model info — keys are prefixed with family name
        model_info = data.get("model_info", {})
        for key, val in model_info.items():
            if "context_length" in key:
                profile.context_length = val
            elif "block_count" in key and "vision" not in key:
                profile.block_count = val
            elif "vision" in key and "block_count" in key:
                profile.supports_vision = True

        # Template inspection — the source of truth for capabilities
        template = data.get("template", "")
        tmpl_lower = template.lower()

        # Think support: look for think/reasoning blocks in template
        profile.supports_think = (
            ".think" in tmpl_lower
            or "think" in tmpl_lower and "reasoning" in tmpl_lower
            or "IsThinkSet" in template
        )

        # Tool support: look for tool/function handling in template
        profile.supports_tools = (
            "tool_call" in tmpl_lower
            or ".toolcalls" in tmpl_lower
            or "function" in tmpl_lower and "tool" in tmpl_lower
        )

        # Family-based fallbacks — community quantizations often ship with
        # stripped templates (e.g. just "{{ .Prompt }}") that don't contain
        # the think/tool/vision markers the base model actually supports.
        # When template detection finds nothing, use known family capabilities
        # so the UI and pipeline aren't degraded by a repackaging choice.
        _FAMILY_CAPS = {
            "gemma4":       {"think": True, "vision": True},
            "gemma3":       {"vision": True},
            "qwen3":        {"think": True, "tools": True},
            "llama4":       {"think": True, "vision": True, "tools": True},
            "deepseek-r1":  {"think": True},
        }
        family_caps = _FAMILY_CAPS.get(profile.family, {})
        if family_caps:
            if not profile.supports_think and family_caps.get("think"):
                profile.supports_think = True
            if not profile.supports_vision and family_caps.get("vision"):
                profile.supports_vision = True
            if not profile.supports_tools and family_caps.get("tools"):
                profile.supports_tools = True

        logger.info(
            "Model profile for %s: family=%s, params=%s, ctx=%d, layers=%d, "
            "think=%s, tools=%s, vision=%s",
            model_name, profile.family, profile.parameter_size,
            profile.context_length, profile.block_count,
            profile.supports_think, profile.supports_tools, profile.supports_vision,
        )

    except Exception as e:
        logger.warning("Failed to detect profile for %s: %s", model_name, e)
        profile.notes = f"Detection failed: {e}"

    return profile


async def set_active(model_name: str) -> ModelProfile:
    """Detect and set the active model profile."""
    global _active
    _active = await detect(model_name)
    return _active


def active() -> ModelProfile:
    """Get the current active model profile."""
    return _active
