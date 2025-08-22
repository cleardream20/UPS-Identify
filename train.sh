#!/bin/bash

CONFIG="MyConfigs/UpsDataset_KNet.py"
cd mmsegmentation
python tools/train.py $CONFIG
