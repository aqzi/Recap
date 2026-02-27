import logging
import os
import platform
import subprocess

logger = logging.getLogger(__name__)

# Model candidates per speed tier, ordered by quality (best first)
MODEL_TIERS = {
    "fast": ["llama3.2:3b", "phi3:mini", "llama3.2:1b", "qwen2.5:0.5b"],
    "medium": ["gemma2:9b", "llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
    "slow": ["llama3.1:70b", "qwen2.5:32b", "gemma2:27b", "llama3.1:8b"],
}

# Approximate RAM requirements in GB for running each model
MODEL_RAM_REQUIREMENTS = {
    "qwen2.5:0.5b": 2,
    "llama3.2:1b": 4,
    "llama3.2:3b": 4,
    "phi3:mini": 4,
    "qwen2.5:7b": 8,
    "mistral:7b": 8,
    "llama3.1:8b": 8,
    "gemma2:9b": 8,
    "gemma2:27b": 20,
    "qwen2.5:32b": 24,
    "llama3.1:70b": 48,
}


def get_system_ram_gb() -> float:
    """Return total system RAM in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        else:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return mem_bytes / (1024 ** 3)
    except Exception:
        logger.warning("Could not detect system RAM, assuming 8 GB")
        return 8.0


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_nvidia_vram_gb() -> float | None:
    """Return total NVIDIA GPU VRAM in GB, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            # Sum all GPUs, nvidia-smi reports in MiB
            total_mib = sum(int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip())
            return total_mib / 1024
    except (FileNotFoundError, Exception):
        pass
    return None


def get_available_ollama_models() -> list[str]:
    """Return list of locally available Ollama model names."""
    try:
        import ollama
        response = ollama.list()
        return [m.model for m in response.models]
    except Exception:
        logger.warning("Could not query Ollama for available models")
        return []


def _is_model_installed(model: str, available: list[str]) -> bool:
    """Check if an Ollama model is installed locally."""
    model_base = model.split(":")[0]
    return any(model in name or name.startswith(model_base) for name in available)


def recommend_model(speed: str) -> tuple[str, str]:
    """Recommend a local Ollama model based on speed preference and hardware.

    Returns (model_name, explanation). The recommendation is based on all known
    models for the tier — not just installed ones. The explanation notes whether
    the model needs to be pulled first.
    """
    candidates = MODEL_TIERS.get(speed, MODEL_TIERS["medium"])
    ram_gb = get_system_ram_gb()
    vram_gb = get_nvidia_vram_gb()
    available = get_available_ollama_models()

    # Use VRAM if available, otherwise system RAM (Apple Silicon shares memory)
    usable_ram = ram_gb if (is_apple_silicon() or vram_gb is None) else vram_gb
    hw_info = f"{usable_ram:.0f} GB {'unified memory' if is_apple_silicon() else 'RAM'}"

    # Pick the best candidate that fits in available memory
    for model in candidates:
        required = MODEL_RAM_REQUIREMENTS.get(model, 8)
        if required > usable_ram:
            continue

        installed = _is_model_installed(model, available)
        if installed:
            return model, f"Recommended for your hardware ({hw_info}) and '{speed}' speed preference"
        else:
            return model, (
                f"Recommended for your hardware ({hw_info}) and '{speed}' speed preference. "
                f"Not yet installed — run: ollama pull {model}"
            )

    # Nothing fits at all — suggest smallest from tier
    fallback = candidates[-1]
    return fallback, f"Your hardware ({hw_info}) is tight for this tier. Smallest option: {fallback} — run: ollama pull {fallback}"
