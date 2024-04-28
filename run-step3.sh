#!/bin/bash
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 run-step3.py
