from hllm.schema.manifest import (
    Artifact,
    BuildInfo,
    Manifest,
    ModelInfo,
    QuantizationInfo,
    RuntimeInfo,
    TargetInfo,
)


def test_manifest_serializes_public_contract() -> None:
    manifest = Manifest(
        schema_version="1.0",
        model=ModelInfo(
            name="Qwen3-1.7B",
            family="qwen3",
            architecture="Qwen3ForCausalLM",
            source_type="huggingface",
            source_id="Qwen/Qwen3-1.7B",
        ),
        quantization=QuantizationInfo(type="int4", bits=4, group_size=64),
        target=TargetInfo(backend="cann", chip="unknown"),
        runtime=RuntimeInfo(context_length=8192),
        build=BuildInfo(converter_version="0.1.0"),
        artifacts=[Artifact(type="model", path="model/model.om")],
    )

    data = manifest.to_dict()
    assert data["schema_version"] == "1.0"
    assert data["model"]["source_id"] == "Qwen/Qwen3-1.7B"
    assert data["quantization"]["bits"] == 4
    assert data["artifacts"][0]["path"] == "model/model.om"
