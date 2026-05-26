# Granite Quantization Tools

This project now supports two related but different outputs for IBM Granite 4.0
1B Base:

1. A **persistent quantized weight artifact** saved as Hugging Face
   `.safetensors` files.
2. An **in-memory OScaR KV-cache patched Granite model** for runtime generation.

The file-producing path is the one to use when you want a quantized model
directory on disk:

```bash
granite-quantize-weights \
  --output-dir artifacts/granite-4.0-1b-base-int4 \
  --quantization int4_weight_only \
  --group-size 128 \
  --dtype bfloat16 \
  --device-map auto
```

That command writes a Hugging Face-compatible directory containing model config,
tokenizer files, and one or more `.safetensors` weight files.

## What This Produces

The main artifact output looks like this:

```text
artifacts/granite-4.0-1b-base-int4/
  config.json
  generation_config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Depending on `--max-shard-size`, the model may be saved as multiple shard files
instead:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

Both forms are normal Hugging Face `save_pretrained` outputs.

The exporter prints a JSON manifest to stdout:

```json
{
  "model_id": "ibm-granite/granite-4.0-1b-base",
  "output_dir": "/absolute/path/artifacts/granite-4.0-1b-base-int4",
  "quantization": "int4_weight_only",
  "group_size": 128,
  "dtype": "bfloat16",
  "device_map": "auto",
  "max_shard_size": "10GB",
  "safetensors_files": [
    {
      "path": "model.safetensors",
      "size_bytes": 123456789
    }
  ]
}
```

The numbers above are examples. Your file sizes depend on the quantization
method and shard size.

## Important Concept

OScaR-KV-Quant and `.safetensors` weight quantization are not the same thing.

OScaR-KV-Quant changes how the **runtime KV cache** is stored during generation.
KV cache tensors are created from your prompt and generated tokens, so they are
not part of the model checkpoint.

The `.safetensors` exporter changes the **model weights** and saves those
weights to disk. This project uses Hugging Face Transformers plus TorchAO for
that file-producing path.

In short:

- Use `granite-quantize-weights` when you want `.safetensors` files.
- Use `load_oscar_patched_granite(...)` when you want runtime OScaR KV-cache
  quantization.
- Treat combining saved weight quantization and OScaR KV-cache quantization as
  an advanced follow-up that should be tested for your exact hardware and
  Transformers/TorchAO versions.

## Baseline Model

This repo is baselined on:

- Model: `ibm-granite/granite-4.0-1b-base`
- Hugging Face architecture: `GraniteMoeHybridForCausalLM`
- Attention class: `GraniteMoeHybridAttention`
- Python: 3.12+

The OScaR runtime path also keeps compatibility hooks for older transformer-style
Granite attention exposed as `GraniteAttention`.

## Requirements

You need:

- Python 3.12+
- `git`
- Enough disk space for the original Granite model and the saved quantized
  artifact
- `torch`, `transformers`, `torchao`, and `safetensors`
- Optional: a Hugging Face token if model downloads require authentication in
  your environment

For the OScaR runtime path, use a Linux machine with an NVIDIA CUDA GPU because
upstream OScaR builds CUDA extensions.

The `.safetensors` weight artifact path uses TorchAO. Hardware support depends
on the selected TorchAO quantization method and your installed PyTorch/TorchAO
versions.

## Setup For `.safetensors` Output

From a fresh shell:

```bash
cd /Users/suneel.marti/opensourceprojects/oscar-granite-kv-quant
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[artifact]"
```

If you use `uv`:

```bash
cd /Users/suneel.marti/opensourceprojects/oscar-granite-kv-quant
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[artifact]"
```

Check that this package imports:

```bash
python -c "from granite_oscar_quant import ArtifactQuantizationConfig; print(ArtifactQuantizationConfig())"
```

That command does not download Granite. It only confirms that your Python
environment can see this package.

## Optional Hugging Face Login

If model download fails with an authentication or gated-model error, log in:

```bash
huggingface-cli login
```

Then rerun the command that failed. Hugging Face will cache downloaded model
files on your machine, so the first run is usually the slowest.

## Create Quantized `.safetensors`

Recommended first artifact run:

```bash
granite-quantize-weights \
  --output-dir artifacts/granite-4.0-1b-base-int4 \
  --quantization int4_weight_only \
  --group-size 128 \
  --dtype bfloat16 \
  --device-map auto \
  --max-shard-size 10GB
```

Available quantization methods:

- `int4_weight_only`
- `int8_weight_only`
- `int8_dynamic_activation_int8_weight`

The default is `int4_weight_only`.

## Load The Saved Artifact

After export, load the saved model directory with Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

artifact_dir = "artifacts/granite-4.0-1b-base-int4"

tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForCausalLM.from_pretrained(
    artifact_dir,
    device_map="auto",
    torch_dtype="auto",
)
```

TorchAO must be installed in the environment that loads the quantized artifact.

## Python API For `.safetensors`

```python
from granite_oscar_quant import (
    ArtifactQuantizationConfig,
    quantize_granite_to_safetensors,
)

report = quantize_granite_to_safetensors(
    ArtifactQuantizationConfig(
        output_dir="artifacts/granite-4.0-1b-base-int4",
        quantization="int4_weight_only",
        group_size=128,
        dtype="bfloat16",
        device_map="auto",
        max_shard_size="10GB",
    )
)

print(report.output_dir)
print(report.safetensors_files)
```

`quantize_granite_to_safetensors(...)` returns a `QuantizedArtifactReport`
Pydantic object that lists the saved `.safetensors` files.

## OScaR Runtime KV-Cache Path

Install the OScaR runtime dependencies only if you also want runtime KV-cache
quantization:

```bash
bash scripts/install_oscar_dependency.sh
```

The script clones upstream OScaR into `third_party/OScaR-KV-Quant`, initializes
its submodules, installs the CUDA PyTorch wheel used by OScaR, and installs
OScaR editable into the active environment.

Run one generation request through the OScaR-patched model:

```bash
granite-oscar-generate \
  --prompt "Explain KV-cache quantization in one paragraph." \
  --max-new-tokens 128 \
  --k-bits 2 \
  --v-bits 2
```

Expected stderr:

```text
patched_granite_attention_layers=<positive integer>
```

Expected stdout:

```text
<generated text from Granite>
```

## Python API For OScaR Runtime Patching

```python
from granite_oscar_quant import OscarKVConfig, load_oscar_patched_granite

patched_granite = load_oscar_patched_granite(
    kv_config=OscarKVConfig(k_bits=2, v_bits=2),
    torch_dtype="auto",
    device_map="auto",
)

# This is an in-memory OScaR KV-patched Granite model object.
model = patched_granite.model
tokenizer = patched_granite.tokenizer

text = patched_granite.generate_text(
    "Explain KV-cache quantization in one sentence.",
    max_new_tokens=64,
)
print(text)
```

This path does not save quantized weights. It patches runtime attention/cache
behavior.

## Baseline Benchmark

Run a before/after comparison between vanilla generation and OScaR KV-cache
quantized generation:

```bash
granite-oscar-baseline \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --k-bits 2 \
  --v-bits 2
```

The command prints JSON with latency, generated tokens, tokens/sec, generated
text, and CUDA peak memory when CUDA is available.

## Important CLI Options

For `.safetensors` artifacts:

- `--output-dir`: where to save the quantized model directory.
- `--quantization`: `int4_weight_only`, `int8_weight_only`, or
  `int8_dynamic_activation_int8_weight`.
- `--group-size`: group size for INT4 weight-only quantization.
- `--dtype`: one of `auto`, `bfloat16`, `float16`, or `float32`.
- `--device-map`: passed to Hugging Face model loading. Defaults to `auto`.
- `--max-shard-size`: passed to `save_pretrained`. Use a large value like
  `10GB` if you want a single `model.safetensors` when possible.

For OScaR runtime generation:

- `--k-bits` and `--v-bits`: key/value cache quantization bit widths.
- `--k-groupsize` and `--v-groupsize`: KV-cache quantization group sizes.
- `--max-new-tokens`: number of tokens to generate.
- `--temperature`: `0.0` means greedy decoding; values above zero enable
  sampling.

## How The Two Paths Work

Weight artifact path:

1. Load Granite with Transformers.
2. Apply TorchAO weight quantization through `TorchAoConfig`.
3. Save with `model.save_pretrained(..., safe_serialization=True)`.
4. Save tokenizer files with `tokenizer.save_pretrained(...)`.
5. Print a report listing generated `.safetensors` files.

OScaR runtime path:

1. Load Granite with Transformers.
2. Find supported Granite attention modules.
3. Replace each attention module's `forward` method with an OScaR-aware eager
   attention implementation.
4. During generation, quantize the runtime KV cache.
5. Keep using normal Hugging Face `model.generate(...)`.

## Project Layout

```text
src/granite_oscar_quant/
  __init__.py       Public package exports
  artifact.py       Quantized .safetensors weight artifact exporter
  benchmark.py      Baseline vs OScaR benchmark CLI
  cli.py            Single OScaR generation CLI
  config.py         Pydantic OScaR KV config
  granite_patch.py  Runtime Granite attention patch
  loader.py         High-level patched Granite model loader
  models.py         Default model id
  schemas.py        Pydantic benchmark result schemas

scripts/
  install_oscar_dependency.sh

tests/
  test_config.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torchao'`

Install the artifact extra:

```bash
python -m pip install -e ".[artifact]"
```

### No `.safetensors` files were written

The exporter calls `save_pretrained(..., safe_serialization=True)` and then
checks for `*.safetensors`. If none are found, check whether model saving failed
earlier in the logs and confirm that `safetensors` is installed.

### CUDA or TorchAO quantization fails

Check that your PyTorch, TorchAO, CUDA, and driver versions are compatible. The
TorchAO backend you choose may have hardware-specific requirements.

### Out of memory while exporting

Try:

- Use a smaller model.
- Use `--device-map auto`.
- Use `--dtype float16` or `--dtype bfloat16`.
- Close other GPU workloads.
- Export on a larger GPU machine.

### `ModuleNotFoundError: No module named 'kv_cache_compression'`

This only affects the OScaR runtime path. Install OScaR:

```bash
bash scripts/install_oscar_dependency.sh
```

### `No supported Granite attention modules were found`

This only affects the OScaR runtime path. The loaded model did not contain
`GraniteMoeHybridAttention` or `GraniteAttention` modules. Make sure you are
using a supported Granite model and a recent enough Transformers version.

## Source References

- IBM Granite 4.0 1B Base model card:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base
- Granite 4.0 1B Base config:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base/blob/main/config.json
- Hugging Face TorchAO quantization docs:
  https://huggingface.co/docs/transformers/main/quantization/torchao
- Hugging Face GraniteMoeHybrid docs:
  https://huggingface.co/docs/transformers/en/model_doc/granitemoehybrid
- OScaR-KV-Quant:
  https://github.com/ZunhaiSu/OScaR-KV-Quant
