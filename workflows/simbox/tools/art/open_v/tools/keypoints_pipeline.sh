#!/bin/bash
CONFIG_PATH="/home/shixu/dev_shixu/DataEngine/workflows/simbox/tools/art/open_v/7265/usd/keypoints_config.json"

cd /home/shixu/dev_shixu/DataEngine/workflows/simbox/tools/art/open_v/tools

# 1. rehier
python rehier.py --config $CONFIG_PATH

# 2. select points
python select_keypoint.py --config $CONFIG_PATH

# 3. Transfer keypoints
python transfer_keypoints.py --config $CONFIG_PATH

# 4. Overwrite keypoints
python overwrite_keypoints.py --config $CONFIG_PATH