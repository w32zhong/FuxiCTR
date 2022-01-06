#!/bin/bash
> config.csv
while true; do
	timeout --foreground 20m bash -xe run_all.sh
	echo 'RESTART'
done
