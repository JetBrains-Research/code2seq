#!/bin/bash

train=60
val=20
test=20
dev=false

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-d, --dataset=NAME        specify dataset name"
      echo "--train=VAL               specify a percentage of dataset used as train set"
      echo "--test=VAL                specify a percentage of dataset used as test set"
      echo "--val=VAL                 specify a percentage of dataset used as validation set"
      exit 0
      ;;
    -d|--dataset*)
      shift
      if test $# -gt 0; then
        DATASET_NAME=$1
      else
        echo "no dataset specified"
        exit 1
      fi
      shift
      ;;
    --train*)
      shift
      if test $# -gt 0; then
        train=$1
      else
        echo "no train specified, using default: 60 %"
        exit 1
      fi
      shift
      ;;
    --test*)
      shift
      if test $# -gt 0; then
        test=$1
      else
        echo "no test specified, using default: 20 %"
        exit 1
      fi
      shift
      ;;
    --val*)
      shift
      if test $# -gt 0; then
        val=$1
      else
        echo "no val specified, using default: 20 %"
        exit 1
      fi
      shift
      ;;
    --dev*)
      shift
      dev=true
      shift
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

DATA_DIR=data
DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d "data" ]
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
  wget https://s3.amazonaws.com/code2seq/datasets/java-med-preprocessed.tar.gz -P data/
  echo "Unzip dataset"
  tar -xvzf data/java-med-preprocessed.tar.gz -C data/
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
    if [ ! -f "$DATA_PATH.tgz" ]
    then
      python utils/download_poj_104.py
    fi

    echo "Unzip dataset"
    tar -xvzf data/poj_104.tgz -C data/
    mv ./data/ProgramData ./data/poj_104

    # In the developer mode we leave only several classes
    if [ $dev ]
    then
      find ./data/poj_104/* -type d -name "[2-9]*" -exec rm -rf {} \;
    fi

    # To prepare our dataset for astminer we need to rename all .txt files to .c files
    for file in ./data/poj_104/*/*.txt
    do
      mv "$file" "${file/.txt/.c}"
    done

    # Splitting dataset on train/test/val parts
    sh ./split_dataset.sh -i ./data/poj_104 -o ./data/poj_104_split --train "$train" --test "$test" --val "$val"
    rm -rf ./data/poj_104
    mv ./data/poj_104_split ./data/poj_104
  fi
  echo "Extracting AST using astminer. You need to clone astminer first"
  mkdir ./data/poj_104_parsed
  cd ../astminer
  ./cli.sh code2vec --lang c --project ../code2seq/data/poj_104/train --output ../code2seq/data/poj_104_parsed/train --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
  ./cli.sh code2vec --lang c --project ../code2seq/data/poj_104/test --output ../code2seq/data/poj_104_parsed/test --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
  ./cli.sh code2vec --lang c --project ../code2seq/data/poj_104/val --output ../code2seq/data/poj_104_parsed/val --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
  cd ../code2seq
  for folder in $(find ./data/poj_104_parsed/*/c -type d)
  do
    for file in $folder/*
    do
      mv "$file" "./data/poj_104_parsed/$(basename ${file/.csv/.$(basename $(dirname $folder)).csv})"
    done
    rm -rf $folder
  done
else
  echo "Dataset $DATASET_NAME does not exist"
fi