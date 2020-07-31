#!/bin/bash
# Script - download codeforces dataset from s3
# options:
# $1              specify a percentage of dataset used as train set
# $2              specify a percentage of dataset used as test set
# $3              specify a percentage of dataset used as validation set
# $4              specify if developer mode is on, default: false
# $5              specify if dataset needs to be shuffled, default: false
# $6              specify a path to astminer .jar file
# $7              specify a path to splitiing script

TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
SHUFFLE=$5
ASTMINER_PATH=$6
SPLIT_SCRIPT=$7
DATA_DIR=./data
DATASET_NAME=poj_104

DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

echo "Downloading dataset ${DATASET_NAME}"
if [ -d "$DATA_PATH" ]
then
  echo "$DATA_PATH exists."
else
  if [ ! -f "$DATA_DIR/poj-104-original.tar.gz" ]
  then
    wget https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz -P $DATA_DIR/
  fi

  echo "Unzip dataset"
  # In the developer mode we leave only several classes
  if [ $DEV ]
  then
    echo "Dev mode"
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz" "ProgramData/[1-3]"
    mv "$DATA_DIR"/ProgramData "$DATA_PATH"
  else
    tar -xvzf "$DATA_DIR/poj-104-original.tar.gz" -C $DATA_DIR/
    mv "$DATA_DIR"/ProgramData "$DATA_PATH"
  fi

  # To prepare our dataset for astminer we need to rename all .txt files to .c files
  echo "Renaming files"
  find "$DATA_PATH"/*  -name "*.txt" -type f -exec sh -c 'mv "$0" "${0%.txt}.c"' {} \;
  # Splitting dataset on train/test/val parts
  echo "Splitting on train/test/val"
  sh "$SPLIT_SCRIPT" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART" "$SHUFFLE"
  rm -rf "$DATA_PATH"
  mv "$DATA_PATH"_split "$DATA_PATH"
fi
echo "Extracting paths using astminer. You need to specify the path to .jar in \"ASTMINER_PATH\" variable first"
if [ -d "$DATA_PATH"_parsed ]
then
  rm -rf "$DATA_PATH"_parsed
fi
mkdir "$DATA_PATH"_parsed

java -jar -Xmx2048m $ASTMINER_PATH code2vec --lang c --project "$DATA_PATH"/train --output "$DATA_PATH"_parsed/train --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx2048m $ASTMINER_PATH code2vec --lang c --project "$DATA_PATH"/test --output "$DATA_PATH"_parsed/test --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx2048m $ASTMINER_PATH code2vec --lang c --project "$DATA_PATH"/val --output "$DATA_PATH"_parsed/val --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
for folder in $(find "$DATA_PATH"_parsed/*/c -type d)
do
  for file in "$folder"/*
  do
    mv "$file" "$DATA_PATH"_parsed/"$(basename "${file/.csv/.$(basename "$(dirname "$folder")").csv}")"
  done
  rm -rf "$(dirname "$folder")"
done