from __future__ import annotations

from pathlib import Path

from hllm.pipeline import BuildOptions, BuildState, Stage


class PipelineRunner:
    """Coordinates the conversion stages without embedding vendor commands."""

    def __init__(self, options: BuildOptions) -> None:
        self.state = BuildState(options)
        self.work_dir = options.output_dir / "work"
        self.dist_dir = options.output_dir / "dist"

    def prepare(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        self.state.record(f"work_dir={self.work_dir}")

    def set_source(self, model_dir: Path) -> None:
        self.state.enter(Stage.DOWNLOAD)
        self.state.model_dir = model_dir.resolve()

    def enter(self, stage: Stage) -> None:
        if self.state.model_dir is None:
            raise RuntimeError("source model is not set")
        self.state.enter(stage)

    def require_source(self) -> Path:
        if self.state.model_dir is None:
            raise RuntimeError("source model is not set")
        return self.state.model_dir
