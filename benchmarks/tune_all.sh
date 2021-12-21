#!/bin/bash
set -e
for path in $(find ../../BARS/ctr_prediction/benchmarks/ -name '*.yaml' | grep _avazu_); do
    filename=$(basename $path)
    model_name=$(echo $path | cut -d / -f6)
    if [[ $filename == model_config.yaml || $filename == dataset_config.yaml ]]; then
        continue
    fi
    if [[ $model_name == HFM || $model_name == LorentzFM || $model_name == FLEN || $model_name == DIN || $model_name == FiGNN || $model_name == FFM ]]; then
        continue
    fi

    python run_param_tuner.py --config $path
done
