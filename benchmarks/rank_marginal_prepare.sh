#!/bin/bash

### Sample ###
 cat ../data/mathclicks/test.csv.retrieval | sort -R | head -n 100 > ../data/mathclicks/test.sample.csv

### Generate rank-marginal features ###
head -n 1 ../data/mathclicks/test.csv.retrieval > ../data/mathclicks/test.csv
while read line; do
	for i in {1..100}; do
		echo $line | awk -F, "{OFS=FS}{\$5 = $i; print}"
	done >> ../data/mathclicks/test.csv
done < ../data/mathclicks/test.sample.csv
