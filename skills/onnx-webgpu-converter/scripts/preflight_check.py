#!/usr/bin/env python3
"""Preflight check for ONNX WebGPU model conversion.

Verifies environment, checks if model exists on Hub, detects task,
and reports whether an ONNX version already exists.

Usage:
    python preflight_check.py <model_id>
    python preflight_check.py distilbert-base-uncased-finetuned-sst-2-english
    python preflight_check.py https://huggingface.co/username/model-name
"""

import subprocess
import sys
import re


def extract_model_id(input_str: str) -> str:
    """Extract model ID from a HuggingFace URL or return as-is."""
    patterns = [
        r"huggingface\.co/([^/]+/[^/\s?#]+)",
        r"hf\.co/([^/]+/[^/\s?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1).rstrip("/")
    return input_str.strip().rstrip("/")


def check_command(cmd: str, install_hint: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  MISSING: {cmd} — install with: {install_hint}")
        return False


def check_python_package(package: str, import_name: str = None) -> bool:
    """Check if a Python package is importable."""
    name = import_name or package
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {name}; print({name}.__version__)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  OK: {package} ({version})")
            return True
    except subprocess.TimeoutExpired:
        pass
    print(f"  MISSING: {package}")
    return False


def check_onnx_community(model_id: str) -> str | None:
    """Check if an onnx-community version exists."""
    model_name = model_id.split("/")[-1]
    onnx_id = f"onnx-community/{model_name}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"""
from huggingface_hub import model_info
try:
    info = model_info("{onnx_id}")
    print(f"FOUND: {{info.id}}")
except Exception:
    print("NOT_FOUND")
"""],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if output.startswith("FOUND:"):
            return output.split("FOUND: ")[1]
    except subprocess.TimeoutExpired:
        pass
    return None


def detect_task(model_id: str) -> str | None:
    """Try to auto-detect the model task."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"""
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("{model_id}")
    arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "Unknown"
    print(f"ARCH: {{arch}}")
    print(f"TYPE: {{config.model_type}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("  Timeout detecting model info")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python preflight_check.py <model_id_or_url>")
        sys.exit(1)

    raw_input = sys.argv[1]
    model_id = extract_model_id(raw_input)

    print(f"\n{'='*60}")
    print(f" ONNX WebGPU Conversion Preflight Check")
    print(f"{'='*60}")
    print(f"\nModel: {model_id}")

    # 1. Check environment
    print(f"\n--- Environment ---")
    all_ok = True

    print("\nPython packages:")
    for pkg, imp in [("optimum", "optimum"), ("onnxruntime", "onnxruntime"),
                     ("torch", "torch"), ("transformers", "transformers")]:
        if not check_python_package(pkg, imp):
            all_ok = False

    print("\nCLI tools:")
    try:
        result = subprocess.run(
            ["optimum-cli", "export", "onnx", "--help"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("  OK: optimum-cli export onnx")
        else:
            print("  MISSING: optimum-cli — pip install 'optimum[onnx]'")
            all_ok = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  MISSING: optimum-cli — pip install 'optimum[onnx]'")
        all_ok = False

    # 2. Check for existing ONNX version
    print(f"\n--- Existing ONNX Versions ---")
    onnx_model = check_onnx_community(model_id)
    if onnx_model:
        print(f"\n  FOUND pre-converted: {onnx_model}")
        print(f"  URL: https://huggingface.co/{onnx_model}")
        print(f"\n  You may not need to convert! Try using this directly:")
        print(f'  pipeline("task", "{onnx_model}", {{ device: "webgpu" }})')
    else:
        print(f"  No onnx-community version found for {model_id.split('/')[-1]}")
        print(f"  Conversion will be needed.")

    # 3. Detect model info
    print(f"\n--- Model Info ---")
    detect_task(model_id)

    # 4. Summary
    print(f"\n--- Summary ---")
    if not all_ok:
        print("\n  Some dependencies are missing. Install with:")
        print('  pip install "optimum[onnx]" onnxruntime torch transformers')
    else:
        print("\n  Environment is ready for conversion!")
        print(f"\n  Quick convert command:")
        print(f"  optimum-cli export onnx --model {model_id} ./{model_id.split('/')[-1]}_onnx/")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
