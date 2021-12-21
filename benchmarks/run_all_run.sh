#!/bin/bash
trap 'echo Interrupted at $(date)' INT
while true; do
	timeout 20m bash -xe all_run.sh
done
