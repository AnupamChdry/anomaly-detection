#!/bin/bash
# Activates the isolated monitoring environment, runs the report
# generator, then returns automatically. Needed because evidently==0.4.25
# requires numpy<2.0, incompatible with the main training/serving
# environment's numpy 2.x (required by torch/anomalib).

source monitoring-env/Scripts/activate
python -m src.monitoring.generate_report "$@"
deactivate