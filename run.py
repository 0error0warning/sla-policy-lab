#!/usr/bin/env python3
"""Start with one BLAS thread for fast, repeatable small-matrix training."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from sla_demo.server import main
main()
