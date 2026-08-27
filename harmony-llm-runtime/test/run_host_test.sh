#!/usr/bin/env bash
# Build and run the host-side runtime import test (no OHOS/NAPI dependency).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
CXX="${CXX:-g++}"
SRC_DIR="$ROOT/native"

echo "== generate test .hllm =="
python3 test/make_test_hllm.py

echo "== build host import test =="
"$CXX" -std=c++17 -I "$SRC_DIR" \
  test/host_import_test.cpp \
  "$SRC_DIR/model_package/ZipReader.cpp" \
  "$SRC_DIR/model_package/Inflate.cpp" \
  "$SRC_DIR/model_package/Json.cpp" \
  "$SRC_DIR/model_package/Sha256.cpp" \
  "$SRC_DIR/model_package/PackageReader.cpp" \
  -o /tmp/host_import_test

echo "== run =="
/tmp/host_import_test test/test_model.hllm

rm -rf test/test_model.hllm.d
echo "OK"
