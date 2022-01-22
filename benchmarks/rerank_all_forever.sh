#!/bin/bash
while true; do
	timeout --foreground 20m bash rerank_all.sh
	echo 'RESTART'
done
