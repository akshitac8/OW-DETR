#!/bin/bash

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/OWOD_new_split.sh
