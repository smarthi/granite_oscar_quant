"""Quantized `.safetensors` artifact export for supported model profiles.

OScaR KV-cache quantization is a runtime generation behavior. A quantized
`.safetensors` file, by contrast, is a persistent weight artifact. This module
adds that second path explicitly by using Hugging Face Transformers plus
TorchAO weight quantization and `save_pretrained(..., safe_serialization=True)`.

The exporter is intentionally model-profile based. Granite and Gemma share the
same quantize/save/report flow, while profiles define the model id and
Transformers auto class needed for each family.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    DEFAULT_ARTIFACT_PROFILE_NAME,
    MODEL_PROFILES,
    ModelAutoClass,
    ModelProfileName,
    resolve_model_profile,
)

ArtifactQuantizationMethod = Literal[
    "int4_weight_only",
    "int8_weight_only",
    "int8_dynamic_activation_int8_weight",
]


class ArtifactQuantizationConfig(BaseModel):
    """Configuration for exporting a quantized model weight artifact.

    What it does:
        Captures the model profile, optional model id override, output
        directory, TorchAO quantization method, dtype, device map, shard size,
        and Hugging Face loading flags used to create a saved quantized model
        directory.

    Why it exists:
        Producing `.safetensors` weights is a different job from OScaR KV-cache
        patching. A separate config keeps that artifact-producing path explicit
        and makes Granite/Gemma differences profile-driven.

    How it helps:
        The CLI and Python API share one validated contract, and the report can
        show exactly which model profile and weight quantization settings
        produced the files.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    profile: ModelProfileName = DEFAULT_ARTIFACT_PROFILE_NAME
    model_id: Optional[str] = None
    auto_model_class: Optional[ModelAutoClass] = None
    output_dir: Optional[Path] = None
    quantization: ArtifactQuantizationMethod = "int4_weight_only"
    group_size: int = Field(default=128, gt=0)
    dtype: str = "auto"
    device_map: Optional[str] = "auto"
    max_shard_size: str = "10GB"
    trust_remote_code: bool = False

    def resolved_profile(self):
        """Return the concrete model profile after applying overrides.

        What it does:
            Combines `profile`, optional `model_id`, and optional
            `auto_model_class` into one `ModelProfile`.

        Why it exists:
            Profile resolution is needed in multiple places: loading, output
            path defaults, and report generation.

        How it helps:
            The rest of the exporter can work against one object with concrete
            model-loading decisions.
        """
        return resolve_model_profile(
            self.profile,
            model_id=self.model_id,
            auto_model_class=self.auto_model_class,
        )

    def resolved_output_dir(self) -> Path:
        """Return the final artifact output directory.

        What it does:
            Uses the explicit `output_dir` when provided, otherwise derives a
            profile-specific default such as `artifacts/gemma-4-e2b-int4`.

        Why it exists:
            The CLI should work with just `--profile gemma4-e2b`, but advanced
            users still need full control over where artifacts are written.

        How it helps:
            Model-specific defaults live in profiles instead of being duplicated
            throughout docs, tests, and code.
        """
        if self.output_dir is not None:
            return self.output_dir
        return self.resolved_profile().default_output_dir(self.quantization)


class SafetensorFile(BaseModel):
    """One `.safetensors` file produced by the artifact exporter.

    What it does:
        Stores the relative path and byte size for a saved safetensors shard.

    Why it exists:
        `save_pretrained` may emit one file or several sharded files depending
        on model size and `max_shard_size`.

    How it helps:
        The export report can tell users exactly which files are the quantized
        weight outputs without requiring them to inspect the directory manually.
    """

    path: str
    size_bytes: int = Field(ge=0)


class QuantizedArtifactReport(BaseModel):
    """JSON-serializable summary of a quantized model export.

    What it does:
        Records the source profile/model, output directory, auto-model class,
        quantization method, dtype, shard size, and produced `.safetensors`
        files.

    Why it exists:
        A quantization run can take time and depends on hardware/software
        choices. The report gives the run a durable manifest that can be saved
        in logs or CI artifacts.

    How it helps:
        Users can immediately see whether the run produced `.safetensors`
        outputs, where they are, and which model family/settings were used.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    profile: ModelProfileName
    model_id: str
    auto_model_class: ModelAutoClass
    output_dir: str
    quantization: ArtifactQuantizationMethod
    group_size: int = Field(gt=0)
    dtype: str
    device_map: Optional[str]
    max_shard_size: str
    safetensors_files: list[SafetensorFile]


def quantize_model_to_safetensors(
    config: ArtifactQuantizationConfig | dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> QuantizedArtifactReport:
    """Quantize model weights and save a `.safetensors` model directory.

    What it does:
        Resolves a model profile, loads the matching model class through
        Transformers with a TorchAO quantization config, saves the quantized
        model with safe serialization, saves model assets, and returns a report
        listing the produced `.safetensors` files.

    Why it exists:
        Granite and Gemma should not need separate exporter implementations.
        Their differences are model id and auto class; the actual TorchAO
        quantize/save/report workflow is shared.

    How it helps:
        A single function creates local Hugging Face-compatible quantized model
        directories for every supported profile.
    """
    resolved = (
        ArtifactQuantizationConfig()
        if config is None
        else ArtifactQuantizationConfig.model_validate(config)
    )
    profile = resolved.resolved_profile()
    output_dir = resolved.resolved_output_dir().expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_quantized_model(profile, resolved, model_kwargs)
    assets = _load_model_assets(profile, resolved)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=resolved.max_shard_size,
    )
    assets.save_pretrained(output_dir)

    safetensors_files = _list_safetensors(output_dir)
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files were written to {output_dir}")

    return QuantizedArtifactReport(
        profile=profile.name,
        model_id=profile.model_id,
        auto_model_class=profile.auto_model_class,
        output_dir=str(output_dir),
        quantization=resolved.quantization,
        group_size=resolved.group_size,
        dtype=resolved.dtype,
        device_map=resolved.device_map,
        max_shard_size=resolved.max_shard_size,
        safetensors_files=safetensors_files,
    )


def quantize_granite_to_safetensors(
    config: ArtifactQuantizationConfig | dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> QuantizedArtifactReport:
    """Backward-compatible Granite artifact export wrapper.

    What it does:
        Delegates to `quantize_model_to_safetensors`, defaulting to the Granite
        profile when no config is supplied.

    Why it exists:
        Earlier versions exposed a Granite-specific function name. Keeping it
        avoids breaking notebooks or scripts while the implementation becomes
        model-profile based.

    How it helps:
        Existing Granite callers keep working, and new Gemma callers can use the
        generic `quantize_model_to_safetensors` API.
    """
    return quantize_model_to_safetensors(config, **model_kwargs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for quantized `.safetensors` export.

    What it does:
        Parses artifact-export flags, runs `quantize_model_to_safetensors`, and
        prints the resulting JSON report.

    Why it exists:
        Beginners often want one command that creates files on disk. The CLI
        provides that path without requiring a Python script.

    How it helps:
        The command's stdout is a machine-readable manifest of the saved
        quantized model artifact for Granite, Gemma, or future profiles.
    """
    args = _parse_args(argv)
    report = quantize_model_to_safetensors(
        ArtifactQuantizationConfig(
            profile=args.profile,
            model_id=args.model_id,
            auto_model_class=args.auto_model_class,
            output_dir=args.output_dir,
            quantization=args.quantization,
            group_size=args.group_size,
            dtype=args.dtype,
            device_map=None if args.device_map == "none" else args.device_map,
            max_shard_size=args.max_shard_size,
            trust_remote_code=args.trust_remote_code,
        )
    )
    print(report.model_dump_json(indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI options for the `.safetensors` artifact exporter.

    What it does:
        Defines the model profile, optional model-id override, destination
        directory, quantization method, dtype/device controls, and serialization
        shard size.

    Why it exists:
        Artifact generation has different knobs than runtime OScaR KV-cache
        generation. Keeping them separate makes the command easier to learn.

    How it helps:
        Parsed arguments become an `ArtifactQuantizationConfig`, so CLI inputs
        and Python API inputs use the same validation.
    """
    parser = argparse.ArgumentParser(
        description="Quantize supported model weights and save a Hugging Face .safetensors artifact."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES),
        default=DEFAULT_ARTIFACT_PROFILE_NAME,
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--auto-model-class",
        choices=["causal-lm", "image-text-to-text"],
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--quantization",
        choices=[
            "int4_weight_only",
            "int8_weight_only",
            "int8_dynamic_activation_int8_weight",
        ],
        default="int4_weight_only",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="10GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _load_quantized_model(
    profile,
    config: ArtifactQuantizationConfig,
    model_kwargs: dict[str, Any],
) -> Any:
    """Load one profile's model with TorchAO quantization enabled.

    What it does:
        Selects the correct Transformers auto-model class for the resolved
        profile and calls `from_pretrained` with dtype, device map, quantization
        config, and any advanced model kwargs.

    Why it exists:
        Gemma4-E2B uses `AutoModelForImageTextToText`, while Granite uses
        `AutoModelForCausalLM`. That difference is the main model-family branch
        in the artifact exporter.

    How it helps:
        The rest of the artifact pipeline can save/report the model without
        caring which auto class loaded it.
    """
    auto_cls = _auto_model_class(profile.auto_model_class)
    return auto_cls.from_pretrained(
        profile.model_id,
        torch_dtype=_torch_dtype(config.dtype),
        device_map=config.device_map,
        quantization_config=_torchao_config(config),
        trust_remote_code=config.trust_remote_code,
        **model_kwargs,
    )


def _load_model_assets(profile, config: ArtifactQuantizationConfig) -> Any:
    """Load tokenizer or processor assets for a model profile.

    What it does:
        Uses `AutoTokenizer` for causal language models and `AutoProcessor` for
        image-text-to-text models, then returns the loaded asset object.

    Why it exists:
        Saved model directories need the matching preprocessing assets. Gemma4
        E2B's multimodal profile may need processor files in addition to text
        tokenizer files.

    How it helps:
        The exported artifact directory can be reloaded with the same Hugging
        Face family APIs that created it.
    """
    if profile.auto_model_class == "image-text-to-text":
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Gemma4-E2B export requires a Transformers version that provides "
                "AutoProcessor for image-text-to-text models."
            ) from exc
        return AutoProcessor.from_pretrained(
            profile.model_id,
            trust_remote_code=config.trust_remote_code,
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        profile.model_id,
        trust_remote_code=config.trust_remote_code,
    )


def _auto_model_class(auto_model_class: ModelAutoClass) -> Any:
    """Return the Transformers auto class for a profile.

    What it does:
        Maps this project's small auto-class enum to concrete Transformers
        classes.

    Why it exists:
        Model profiles should not store imported Python classes because that
        would make lightweight config imports pull in Transformers immediately.

    How it helps:
        Imports stay lazy, and missing Transformers support produces an error at
        the point where a user actually tries to export that profile.
    """
    if auto_model_class == "causal-lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    if auto_model_class == "image-text-to-text":
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "Gemma4-E2B export requires a Transformers version with "
                "AutoModelForImageTextToText support."
            ) from exc
        return AutoModelForImageTextToText
    raise ValueError(f"Unsupported auto model class: {auto_model_class}")


def _list_safetensors(output_dir: Path) -> list[SafetensorFile]:
    """List saved `.safetensors` files below an artifact directory.

    What it does:
        Recursively finds `*.safetensors` files and records their relative paths
        and sizes.

    Why it exists:
        `save_pretrained` may emit a single file or multiple shards depending
        on model size and shard settings.

    How it helps:
        The export report can identify all persistent weight outputs in a stable
        JSON shape.
    """
    return [
        SafetensorFile(
            path=str(path.relative_to(output_dir)),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(output_dir.rglob("*.safetensors"))
    ]


def _torchao_config(config: ArtifactQuantizationConfig) -> Any:
    """Create the Transformers TorchAO quantization config.

    What it does:
        Maps this project's simple quantization method names to
        `transformers.TorchAoConfig` arguments.

    Why it exists:
        Transformers owns the public `quantization_config` integration point
        and delegates to TorchAO internally. The CLI should not make beginners
        construct those objects manually just to create a `.safetensors`
        artifact.

    How it helps:
        The artifact exporter can support a small stable set of quantization
        choices while preserving access to official Transformers/TorchAO
        serialization behavior.
    """
    quant_dict = {
        "int4_weight_only": {"group_size": config.group_size},
        "int8_weight_only": {},
        "int8_dynamic_activation_int8_weight": {},
    }
    kwargs = quant_dict[config.quantization]
    if kwargs is None:
        raise ValueError(f"Unsupported quantization method: {config.quantization}")

    try:
        from transformers import TorchAoConfig

        return TorchAoConfig(config.quantization, **kwargs)
    except Exception as exc:
        raise ImportError(
            "TorchAO artifact export requires `torchao>=0.15` and a Transformers "
            "version with TorchAoConfig support. Install this project with the "
            "artifact extra or run `python -m pip install torchao>=0.15`."
        ) from exc


def _torch_dtype(name: str) -> Any:
    """Map a dtype string to the value expected by Transformers loading.

    What it does:
        Converts friendly CLI dtype names into torch dtype objects, preserving
        `"auto"` for Transformers' automatic dtype selection.

    Why it exists:
        Artifact export should be scriptable from the command line while still
        passing proper dtype values into `from_pretrained`.

    How it helps:
        Users can choose common loading precisions without writing Python.
    """
    if name == "auto":
        return "auto"

    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


if __name__ == "__main__":
    raise SystemExit(main())
