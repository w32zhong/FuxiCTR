# FuxiCTR
Test run:
```sh
cd benchmarks
python run_expid.py --expid MathClicks_test
```

Replace `data/mathclicks/{train,valid,test}.csv` with real-world data.

Run all experiment:
```
python run_all.sh
```

Run parameter grid search (download [the BARS repo](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks) ahead of time):
```
python run_param_tuner.py --config ../../BARS/ctr_prediction/benchmarks/WideDeep/WideDeep_avazu_x4_001/WideDeep_avazu_x4_tuner_config_01.yaml
```

Create a list of commands for parameter grid search, and run them:
```
./tune_all.sh > all_run.sh
rm -rf touch
./run_all_run.sh
```
