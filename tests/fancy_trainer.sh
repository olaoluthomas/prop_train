#!/bin/bash

usage() {
    echo "incorrect or no argments supplied.

usage: run_trainer.sh 

Supply segment to train

Must be one of [104, 106, 108, 109, 110, 112, 114, 115, 116, 'prospects']"
}

TABLENAME="dm_pc_tiny_data"

echo "Segment to train: "
read SEGMENT

if [ $SEGMENT == 'prospects' ]; then
    python -m proptrainer.task --table_name=${TABLENAME} --segment=${SEGMENT}
    ./cleanup.sh
elif [ $SEGMENT -eq 104 ] || [ $SEGMENT -eq 106 ] || [ $SEGMENT -eq 108 ] || [ $SEGMENT -eq 109 ] || [ $SEGMENT -eq 110 ] || [ $SEGMENT -eq 112 ] || [ $SEGMENT -eq 114 ] || [ $SEGMENT -eq 115 ] || [ $SEGMENT -eq 116 ]; then
    python -m proptrainer.task --table_name=${TABLENAME} --segment=${SEGMENT}
    ./cleanup.sh
else 
    usage
    exit 1
fi