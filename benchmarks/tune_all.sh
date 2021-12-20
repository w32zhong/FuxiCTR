#!/bin/bash
set -e
for path in $(find ../../BARS/ctr_prediction/benchmarks/ -name '*.yaml' | grep _avazu_); do
    model_name=$(echo $path | cut -d / -f6)
    if [[ $model_name == HFM || $model_name == LorentzFM || $model_name == FLEN || $model_name == DIN || $model_name == FiGNN || $model_name == FFM ]]; then
        continue
    fi

    python run_param_tuner.py --config $path
done
