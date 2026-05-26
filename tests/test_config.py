"""Lightweight tests for config validation and report schemas.

These tests avoid loading torch models so they can run quickly on a developer
machine while still protecting the public configuration and benchmark JSON
contracts that the heavier Granite/OScaR flow depends on.
"""

import json

from pydantic import ValidationError

from granite_oscar_quant.artifact import (
    ArtifactQuantizationConfig,
    QuantizedArtifactReport,
    SafetensorFile,
)
from granite_oscar_quant.config import OscarKVConfig
from granite_oscar_quant.loader import OscarPatchedGraniteModel
from granite_oscar_quant.models import DEFAULT_GRANITE_MODEL_ID
from granite_oscar_quant.schemas import BenchmarkReport, BenchmarkRun


def test_default_config_validates_and_exports_namespace():
    """Verify the default OScaR config is valid and namespace-compatible.

    The adapter passes a namespace to upstream OScaR, so this test protects both
    the Pydantic defaults and the compatibility shim used by the patcher.
    """
    config = OscarKVConfig()

    namespace = config.as_namespace()

    assert namespace.k_bits == 2
    assert namespace.v_bits == 2
    assert namespace.k_groupsize == 32
    assert namespace.k_norm_factoring is True
    assert config.model_dump()["v_groupsize"] == 32


def test_invalid_bits_raise():
    """Verify Pydantic rejects unsupported quantization bit widths.

    Catching this before model loading keeps invalid CLI/API input from wasting
    GPU memory or failing deep inside OScaR's CUDA-backed path.
    """
    try:
        OscarKVConfig(k_bits=1)
    except ValidationError as exc:
        assert "greater than or equal to 2" in str(exc)
    else:
        raise AssertionError("Expected invalid bit width to raise")


def test_default_model_focuses_granite_40_1b_base():
    """Verify every default path stays baselined on Granite 4.0 1B Base.

    The model id is shared by the CLI, README examples, and benchmark tests, so
    one assertion helps catch accidental drift back to older Granite defaults.
    """
    assert DEFAULT_GRANITE_MODEL_ID == "ibm-granite/granite-4.0-1b-base"


def test_benchmark_report_serializes_as_json():
    """Verify benchmark Pydantic models emit stable JSON.

    The baseline CLI writes `BenchmarkReport` JSON to stdout, so this checks the
    schema shape without requiring a model download or generation run.
    """
    report = BenchmarkReport(
        model_id=DEFAULT_GRANITE_MODEL_ID,
        prompt_tokens=5,
        k_bits=2,
        v_bits=2,
        runs=[
            BenchmarkRun(
                label="baseline",
                elapsed_seconds=1.25,
                new_tokens=10,
                tokens_per_second=8.0,
                text=" Paris.",
            )
        ],
    )

    payload = json.loads(report.model_dump_json())

    assert payload["model_id"] == DEFAULT_GRANITE_MODEL_ID
    assert payload["runs"][0]["label"] == "baseline"


def test_patched_granite_wrapper_identifies_model_output():
    """Verify the loader output schema represents the patched model object.

    The real loader downloads weights and applies OScaR. This lightweight test
    uses placeholder objects to protect the shape of the returned wrapper
    without requiring a model download.
    """
    wrapped = OscarPatchedGraniteModel(
        model_id=DEFAULT_GRANITE_MODEL_ID,
        model=object(),
        tokenizer=object(),
        kv_config=OscarKVConfig(),
        patched_attention_layers=24,
    )

    assert wrapped.model_id == DEFAULT_GRANITE_MODEL_ID
    assert wrapped.patched_attention_layers == 24
    assert wrapped.kv_config.k_bits == 2


def test_artifact_config_defaults_to_quantized_safetensors_output():
    """Verify the artifact path defaults to Granite 4 INT4 weight export.

    The `.safetensors` output path is separate from OScaR runtime KV-cache
    patching, so this test protects the default weight artifact config.
    """
    config = ArtifactQuantizationConfig()

    assert config.model_id == DEFAULT_GRANITE_MODEL_ID
    assert config.quantization == "int4_weight_only"
    assert config.group_size == 128


def test_quantized_artifact_report_lists_safetensor_files():
    """Verify the artifact report identifies saved `.safetensors` shards.

    The exporter may produce one file or many, depending on shard size. This
    lightweight test protects the JSON shape without loading Granite.
    """
    report = QuantizedArtifactReport(
        model_id=DEFAULT_GRANITE_MODEL_ID,
        output_dir="/tmp/granite-int4",
        quantization="int4_weight_only",
        group_size=128,
        dtype="auto",
        device_map="auto",
        max_shard_size="10GB",
        safetensors_files=[
            SafetensorFile(path="model.safetensors", size_bytes=1234),
        ],
    )

    payload = json.loads(report.model_dump_json())

    assert payload["safetensors_files"][0]["path"] == "model.safetensors"
    assert payload["quantization"] == "int4_weight_only"
