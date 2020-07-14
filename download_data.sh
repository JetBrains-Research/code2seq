#!/bin/bash

DATASET_NAME=$1
DATA_DIR=data
DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d "$data" ]
then
  mkdir data
fi

if [ "$DATASET_NAME" = "java-small" ]
then
  echo "Downloading dataset $1"
  wget https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz -P data/
  echo "Unzip dataset"
  tar -xvzf data/java-small-preprocessed.tar.gz -C data/
  echo "Deleting .tar.gz"
  rm data/java-small-preprocessed.tar.gz
elif [ "$DATASET_NAME" = "java-medium" ]
then
  echo "Downloading dataset $1"
  wget https://s3.amazonaws.com/code2seq/datasets/java-medium-preprocessed.tar.gz -P data/
  echo "Unzip dataset"
  tar -xvzf data/java-medium-preprocessed.tar.gz -C data/
  echo "Deleting .tar.gz"
  rm data/java-medium-preprocessed.tar.gz
elif [ "$DATASET_NAME" = "java-large" ]
then
  echo "Downloading dataset $1"
  wget https://s3.amazonaws.com/code2seq/datasets/java-large-preprocessed.tar.gz -P data/
  echo "Unzip dataset"
  tar -xvzf data/java-large-preprocessed.tar.gz -C data/
  echo "Deleting .tar.gz"
  rm data/java-large-preprocessed.tar.gz
elif [ "$DATASET_NAME" = "poj_104" ]
then
  echo "Downloading dataset $1"
  if [ -d "$DATA_PATH" ]
  then
    echo "$DATA_PATH exists."
  else
    python utils/download_poj_104.py
    echo "Unzip dataset"
    tar -xvzf data/poj_104.tgz -C data/
    mv ./data/ProgramData ./data/poj_104
    for file in ./data/poj_104/*/*.txt
    do
      mv "$file" "${file/.txt/.c}"
    done
  fi
  echo "Extracting AST using astminer. You need to clone astminer first"
  mkdir ./data/poj_104_parsed
  cd ../astminer
  ./cli.sh code2vec --lang c --project ../code2seq/data/poj_104 --output ../code2seq/data/poj_104_parsed --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
else
  echo "Dataset $DATASET_NAME does not exist"
fi