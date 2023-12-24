#!/bin/bash

# In termimal
# e.g bash run.sh <image_name> <COMET_API_KEY>
# v5q8O8LftZtvOcoXlVM8Ku8fH
docker run -p 5000:5000 -e COMET_API_KEY="$2" $1
# docker run -p 5000:5000 -e COMET_API_KEY="${COMET_API_KEY}" $1
