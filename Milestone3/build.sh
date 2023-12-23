#!/bin/bash

# In terminal:
# e.g bash build.sh <image_name>
docker build -f Dockerfile.serving -t $1 ./