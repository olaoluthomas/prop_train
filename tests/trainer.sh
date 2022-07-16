#!/bin/bash
usage() {
    echo "incorrect or no argments supplied.

usage: trainer.sh 

Supply segment to train

Must be one of [104, 106, 108, 109, 110, 112, 114, 115, 116, 'prospects']"
}

SEGMENT=$1
TABLENAME="dm_pc_tiny_data"
echo "Training segment ${SEGMENT}..."
python -m proptrainer.task --segment=${SEGMENT} --table_name=${TABLENAME}

./cleanup.sh