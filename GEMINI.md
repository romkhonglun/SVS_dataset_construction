# Project Instructions

## Environment Management
- **Mandatory Tool:** Always use `pixi` for environment management, dependency installation, and running scripts in this project.
- **Commands:**
  - To install dependencies: `pixi install`
  - To run a script: `pixi run python <script_name>.py` or `pixi run <task_name>` if defined in `pyproject.toml`.

## Hardware Acceleration
- Use `onnxruntime-gpu` with `CUDAExecutionProvider` for all ONNX model inferences (e.g., MIDI extraction).
