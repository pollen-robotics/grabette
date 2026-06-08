"""Regenerate gRPC Python files from proto/gripper.proto.

After generation, fixes the import in _grpc.py to use relative imports
(grpc_tools generates absolute imports that don't work inside a package).
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("gripette/proto")

def main():
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        "--proto_path=proto",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",
        "gripper.proto",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Fix absolute import → relative import in _grpc.py
    grpc_file = OUTPUT_DIR / "gripper_pb2_grpc.py"
    text = grpc_file.read_text()
    text = text.replace(
        "import gripper_pb2 as gripper__pb2",
        "from . import gripper_pb2 as gripper__pb2",
    )
    grpc_file.write_text(text)

    print("Proto files generated and fixed in gripette/proto/")

if __name__ == "__main__":
    main()
