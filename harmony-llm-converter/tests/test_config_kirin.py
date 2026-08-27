from pathlib import Path

from hllm.config import load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_kirinx90_profile_loads_omg_fields() -> None:
    profile = load_profile(ROOT / "configs/qwen3-0.6b-kirinx90.example.yaml")
    assert profile.target_chip == "KirinX90"
    assert profile.conversion_tool == "omg"
    assert profile.framework == 5
    assert profile.cann_target == "omc"
