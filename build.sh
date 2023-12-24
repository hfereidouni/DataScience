#!/bin/bash

# In terminal:
# e.g bash build.sh <image_name>
docker build -f Dockerfile.serving -t flask-app ./
# docker build -f Dockerfile.streamlit -t st-app ./