#!/bin/bash
set -euo pipefail
# run me as `memgraph` user

CI=false
CACHE_PRESENT=false
CUDA=false
ARCH=amd64
WHEEL_CACHE_DIR="$(pwd)/wheels"
DEB_PACKAGE=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI=true
      shift
      ;;
    --cache-present)
      CACHE_PRESENT=$2
      shift 2
      ;;
    --cuda)
      CUDA=$2
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --wheel-cache-dir)
      WHEEL_CACHE_DIR=$2
      shift 2
      ;;
    --deb-package)
      DEB_PACKAGE=true
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$CUDA" == true && "$ARCH" != "amd64" ]]; then
    echo "CUDA is only supported on amd64 architecture."
    exit 1
fi

export PIP_BREAK_SYSTEM_PACKAGES=1
if [[ "$CUDA" == true && "$DEB_PACKAGE" == "false" ]]; then
    requirements_file="requirements-gpu.txt"
else
    requirements_file="requirements.txt"
fi

# The requirements file declares every package, DGL and PyTorch Geometric
# included. A CPU build offers its wheel cache as an extra pip source.
if [[ "$CACHE_PRESENT" == "true" && "$CUDA" != true ]]; then
    echo "Using cached torch packages"
    export PIP_FIND_LINKS="$WHEEL_CACHE_DIR"
fi

if [ "$CI" = true ]; then
    # for building the docker image
    python3 -m pip install --no-cache-dir -r "/tmp/${requirements_file}"
    python3 -m pip install --no-cache-dir -r /tmp/auth_module-requirements.txt
elif [[ "$DEB_PACKAGE" == "true" ]]; then
    # for installing python deps during deb package installation
    python3 -m pip install --no-cache-dir -r "/usr/lib/memgraph/mage-requirements.txt"
    python3 -m pip install --no-cache-dir -r "/usr/lib/memgraph/auth_module/requirements.txt"
else
    # for installing locally, from within the mage repo
    python3 -m pip install --no-cache-dir -r "$(pwd)/python/${requirements_file}"
    python3 -m pip install --no-cache-dir -r "$(pwd)/cpp/memgraph/src/auth/reference_modules/requirements.txt"
fi

rm -fr /home/memgraph/.cache/pip
