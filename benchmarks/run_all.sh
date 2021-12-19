#!/bin/bash
set -e
> config.csv
for path in $(find ../config/model_config -name '*.yaml'); do
    filename=$(basename $path)
    if [[ $filename == tests.yaml || $filename == common.yaml ]]; then
        continue
    fi
    model_name=$(echo $filename | cut -d . -f1)
    if [[ $model_name == HFM ]]; then
        continue
    fi
    echo ${model_name}
    python run_expid.py --expid ${model_name}_base
done
