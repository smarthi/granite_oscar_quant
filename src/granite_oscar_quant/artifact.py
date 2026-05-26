"""Quantized `.safetensors` artifact export for IBM Granite.

OScaR KV-cache quantization is a runtime generation behavior. A quantized
`.safetensors` file, by contrast, is a persistent weight artifact. This module
adds that second path explicitly by using Hugging Face Transformers plus
TorchAO weight quantization and `save_pretrained(..., safe_serialization=True)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import DEFAULT_GRANITE_MODEL_ID

ArtifactQuantizationMethod = Literal[
    "int4_weight_only",
    "int8_weight_only",
    "int8_dynamic_activation_int8_weight",
]


class ArtifactQuantizationConfig(BaseModel):
    """Configuration for exporting a quantized Granite weight artifact.

    What it does:
        Captures the model id, output directory, TorchAO quantization method,
        dtype, device map, shard size, and Hugging Face loading flags used to
        create a saved quantized model directory.

    Why it exists:
        Producing `.safetensors` weights is a different job from OScaR KV-cache
        patching. A separate config keeps that artifact-producing path explicit
        and avoids mixing runtime cache settings with persistent weight settings.

    How it helps:
        The CLI and Python API share one validated contract, and the report can
        show exactly which weight quantization settings produced the files.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = DEFAULT_GRANITE_MODEL_ID
    output_dir: Path = Path("artifacts/granite-4.0-1b-base-int4")
    quantization: ArtifactQuantizationMethod = "int4_weight_only"
    group_size: int = Field(default=128, gt=0)
    dtype: str = "auto"
    device_map: Optional[str] = "auto"
    max_shard_size: str = "10GB"
    trust_remote_code: bool = False


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
    """JSON-serializable summary of a quantized Granite export.

    What it does:
        Records the source model, output directory, quantization method, dtype,
        shard size, and produced `.safetensors` files.

    Why it exists:
        A quantization run can take time and depends on hardware/software
        choices. The report gives the run a durable manifest that can be saved
        in logs or CI artifacts.

    How it helps:
        Users can immediately see whether the run produced `.safetensors`
        outputs, where they are, and which quantization settings were used.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    output_dir: str
    quantization: ArtifactQuantizationMethod
    group_size: int = Field(gt=0)
    dtype: str
    device_map: Optional[str]
    max_shard_size: str
    safetensors_files: list[SafetensorFile]


def quantize_granite_to_safetensors(
    config: ArtifactQuantizationConfig | dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> QuantizedArtifactReport:
    """Quantize Granite weights and save a `.safetensors` model directory.

    What it does:
        Loads Granite through Transformers with a TorchAO quantization config,
        saves the quantized model with safe serialization, saves the tokenizer,
        and returns a report listing the produced `.safetensors` files.

    Why it exists:
        Users who want a file artifact need weight quantization, not OScaR's
        runtime KV-cache patch. This function provides that artifact path while
        keeping the OScaR runtime path available elsewhere in the package.

    How it helps:
        A single call creates a local Hugging Face-compatible model directory
        that can be loaded later with `AutoModelForCausalLM.from_pretrained`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = (
        ArtifactQuantizationConfig()
        if config is None
        else ArtifactQuantizationConfig.model_validate(config)
    )
    output_dir = resolved.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved.model_id,
        trust_remote_code=resolved.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved.model_id,
        torch_dtype=_torch_dtype(resolved.dtype),
        device_map=resolved.device_map,
        quantization_config=_torchao_config(resolved),
        trust_remote_code=resolved.trust_remote_code,
        **model_kwargs,
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=resolved.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)

    safetensors_files = [
        SafetensorFile(
            path=str(path.relative_to(output_dir)),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(output_dir.rglob("*.safetensors"))
    ]
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files were written to {output_dir}")

    return QuantizedArtifactReport(
        model_id=resolved.model_id,
        output_dir=str(output_dir),
        quantization=resolved.quantization,
        group_size=resolved.group_size,
        dtype=resolved.dtype,
        device_map=resolved.device_map,
        max_shard_size=resolved.max_shard_size,
        safetensors_files=safetensors_files,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for quantized Granite `.safetensors` export.

    What it does:
        Parses artifact-export flags, runs `quantize_granite_to_safetensors`,
        and prints the resulting JSON report.

    Why it exists:
        Beginners often want one command that creates files on disk. The CLI
        provides that path without requiring a Python script.

    How it helps:
        The command's stdout is a machine-readable manifest of the saved
        quantized model artifact.
    """
    args = _parse_args(argv)
    report = quantize_granite_to_safetensors(
        ArtifactQuantizationConfig(
            model_id=args.model_id,
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
        Defines the source model, destination directory, quantization method,
        dtype/device controls, and serialization shard size.

    Why it exists:
        Artifact generation has different knobs than runtime OScaR KV-cache
        generation. Keeping them separate makes the command easier to learn.

    How it helps:
        Parsed arguments become an `ArtifactQuantizationConfig`, so CLI inputs
        and Python API inputs use the same validation.
    """
    parser = argparse.ArgumentParser(
        description="Quantize Granite weights and save a Hugging Face .safetensors artifact."
    )
    parser.add_argument("--model-id", default=DEFAULT_GRANITE_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _torchao_config(config: ArtifactQuantizationConfig) -> Any:
    """Create the Transformers TorchAO quantization config.

    What it does:
        Maps this project's simple quantization method names to TorchAO config
        objects and wraps them in `transformers.TorchAoConfig`.

    Why it exists:
        TorchAO exposes several Python config classes. The CLI should not make
        beginners import those classes manually just to create a `.safetensors`
        artifact.

    How it helps:
        The artifact exporter can support a small stable set of quantization
        choices while preserving access to official Transformers/TorchAO
        serialization behavior.
    """
    try:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )
        from transformers import TorchAoConfig
    except ImportError as exc:
        raise ImportError(
            "TorchAO artifact export requires `torchao>=0.15` and a Transformers "
            "version with TorchAoConfig support. Install this project with the "
            "artifact extra or run `python -m pip install torchao>=0.15`."
        ) from exc

    if config.quantization == "int4_weight_only":
        quant_type = Int4WeightOnlyConfig(group_size=config.group_size)
    elif config.quantization == "int8_weight_only":
        quant_type = Int8WeightOnlyConfig()
    elif config.quantization == "int8_dynamic_activation_int8_weight":
        quant_type = Int8DynamicActivationInt8WeightConfig()
    else:
        raise ValueError(f"Unsupported quantization method: {config.quantization}")

    return TorchAoConfig(quant_type=quant_type)


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
