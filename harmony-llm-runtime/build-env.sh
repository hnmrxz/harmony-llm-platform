#!/usr/bin/env bash
# Source this to configure the HarmonyOS Command Line Tools + SDK for building
# the runtime on Linux:
#
#   source build-env.sh
#   ohpm install
#   hvigorw --mode project -p product=default -p buildMode=debug assembleApp
#
# Adjust DEVECO_CLT_ROOT to wherever you extracted commandline-tools-linux-x64-*.zip.

DEVECO_CLT_ROOT="${DEVECO_CLT_ROOT:-/home/jat/Develop/deveco-clt/command-line-tools}"
DEVECO_JAVA_ROOT="${DEVECO_JAVA_ROOT:-/home/jat/Develop/java}"

export DEVECO_SDK_HOME="$DEVECO_CLT_ROOT/sdk"
export DEVECO_NODE_HOME="$DEVECO_CLT_ROOT/tool/node"
export PATH="$DEVECO_CLT_ROOT/bin:$DEVECO_NODE_HOME/bin:$PATH"

# Java is required by the final HAP packaging/signing step (hap-sign-tool.jar).
# Point DEVECO_JAVA_ROOT at an extracted Temurin JRE/JDK when present.
if [ -d "$DEVECO_JAVA_ROOT" ]; then
  JAVA_BIN="$(find "$DEVECO_JAVA_ROOT" -path '*/bin/java' -type f 2>/dev/null | head -1)"
  if [ -n "$JAVA_BIN" ]; then
    export JAVA_HOME="$(dirname "$(dirname "$JAVA_BIN")")"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Using Java: $JAVA_BIN"
  else
    echo "WARN: no java under $DEVECO_JAVA_ROOT; HAP packaging will fail (spawn java ENOENT)."
  fi
else
  echo "WARN: DEVECO_JAVA_ROOT=$DEVECO_JAVA_ROOT not found; HAP packaging will fail."
fi

echo "DEVECO_SDK_HOME=$DEVECO_SDK_HOME"
echo "hvigor: $(hvigorw --version 2>/dev/null | head -1)"
