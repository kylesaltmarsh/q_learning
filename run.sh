#!/bin/bash
if [[ "$1" = train ]]
then
    python frozen_lake.py -is_slippery 0 --path /opt/ml/model/
else
    python frozen_lake.py --mode play
fi