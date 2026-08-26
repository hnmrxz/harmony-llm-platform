# HLLM Package Specification v1.0

`.hllm` is the portable delivery artifact between the Ubuntu Converter and HarmonyOS Runtime.

## Container

The package is a ZIP-compatible archive with the `.hllm` extension.

```text
model.hllm
├── manifest.json
└── model/
    ├── <device-specific offline model files>
    ├── tokenizer/
    └── config/
```

The Runtime must never execute arbitrary scripts from the archive.

## Manifest

Required top-level fields:

```json
{
  "schema_version": "1.0",
  "model": {},
  "quantization": {},
  "target": {},
  "runtime": {},
  "build": {},
  "artifacts": []
}
```

### model

```json
{
  "name": "Qwen3.8-27B",
  "family": "qwen3_5",
  "architecture": "Qwen3_5ForConditionalGeneration",
  "source_type": "huggingface",
  "source_id": "Qwen/Qwen3.8-27B",
  "revision": "<commit>"
}
```

### quantization

```json
{
  "type": "cann_4bit",
  "bits": 4,
  "group_size": null
}
```

### target

The target is hardware-specific. Offline models are not assumed to be portable across unrelated AI hardware. This aligns with HarmonyOS NNRt documentation, which describes hardware-specific offline model loading. citeturn624475search2

```json
{
  "backend": "cann",
  "chip": "<validated-kirin-target>",
  "runtime_version": "<validated-cann-version>"
}
```

### runtime

```json
{
  "context_length": 32768,
  "minimum_memory_mb": 0
}
```

### artifacts

Each artifact declares its location and integrity:

```json
{
  "type": "model",
  "path": "model/qwen.om",
  "sha256": "...",
  "size": 123
}
```

## Installation Contract

HarmonyOS Runtime must perform, in order:

1. Open the archive.
2. Parse `manifest.json`.
3. Validate `schema_version`.
4. Validate every declared artifact exists.
5. Validate SHA-256 checksums.
6. Validate target hardware/runtime requirements.
7. Validate storage capacity.
8. Install atomically.
9. Load the device-specific offline model through the native inference layer.

The official HarmonyOS CANN Kit integration documentation describes offline-model loading through the native layer and execution through NNRt APIs such as offline model construction and executor execution. citeturn624475search0

## Text-Only vs Multimodal Models

A model package may represent a multimodal source while targeting a text-only Runtime profile. The Converter must not silently claim vision/video support unless the resulting artifacts and Runtime implement it.

Recommended future manifest field:

```json
{
  "capabilities": {
    "text_generation": true,
    "vision": false,
    "video": false
  }
}
```

## Converter Contract

The Converter must produce a package that is self-contained for Runtime use. Runtime must not require:

- Python
- PyTorch
- Transformers
- Hugging Face credentials
- ONNX tooling
- CANN conversion tools

## Security

`.hllm` is data, not executable code. Runtime must reject path traversal, absolute archive paths, duplicate manifest entries, and undeclared executable content.
