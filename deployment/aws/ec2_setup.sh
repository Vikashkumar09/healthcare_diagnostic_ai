#!/bin/bash

sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

git clone https://github.com/Vikashkumar09/healthcare_diagnostic_ai.git
cd healthcare_diagnostic_ai

docker build -t brain-tumor-ai .
docker run -d -p 8000:8000 brain-tumor-ai


