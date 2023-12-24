#!/bin/bash

# In termimal
# e.g bash run.sh <image_name> <COMET_API_KEY>
# v5q8O8LftZtvOcoXlVM8Ku8fH
docker run -p 8000:8000 -e COMET_API_KEY="v5q8O8LftZtvOcoXlVM8Ku8fH" flask-app
# docker run -p 8080:8080 -e COMET_API_KEY="${COMET_API_KEY}" $1