#!/usr/bin/env bash

echo "**** Creating Python virtualenv ****"
python3 -m venv .venv --prompt triton
source .venv/bin/activate

PYTHON_VERSION=$( python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "wheels-py${PYTHON_VERSION}*" \
    --dir .

cd wheels-py${PYTHON_VERSION}*
pip install torch-* intel_extension_for_pytorch-*