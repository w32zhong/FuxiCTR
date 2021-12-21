#!/bin/bash
while true; do
	timeout --foreground 20m bash -xe all_run.sh
	echo 'RESTART'
done
