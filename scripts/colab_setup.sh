#!/usr/bin/env bash
set -e
apt-get update -qq
apt-get install -y -qq openjdk-17-jdk fonts-nanum
pip install -U -q "transformers>=4.46.0" accelerate bitsandbytes sentencepiece huggingface_hub konlpy JPype1 scipy statsmodels pandas numpy matplotlib tqdm pyyaml psutil
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
