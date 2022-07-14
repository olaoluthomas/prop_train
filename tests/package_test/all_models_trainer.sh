#!/bin/bash

TABLENAME="dm_pc_tiny_data"
seg=( 104 106 108 109 110 112 114 115 116 "prospects" )

for i in "${seg[@]}"
do 
    echo "Training segment ${i}..."
    python -m proptrainer.task --segment=$i --table_name=${TABLENAME}
done

./cleanup.sh