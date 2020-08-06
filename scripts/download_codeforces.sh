#!/bin/bash
# Script - download codeforces dataset from s3
# options:
# $1              specify a percentage of dataset used as train set
# $2              specify a percentage of dataset used as test set
# $3              specify a percentage of dataset used as validation set
# $4              specify if developer mode is on, default: false
# $5              specify a path to astminer .jar file
# $6              specify a path to splitiing script

TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
ASTMINER_PATH=$5
SPLIT_SCRIPT=$6
DATA_DIR=./data
DATASET_NAME=codeforces

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
  if [ ! -f "$DATA_DIR/$DATASET_NAME.zip" ]
  then
    aws s3 cp s3://datasets.ml.labs.aws.intellij.net/codeforces-code-clone/anti-plagiarism-datasets-master.zip "$DATA_DIR/$DATASET_NAME.zip"
  fi

  echo "Unzip dataset"

  if $DEV
  then
    echo "Dev mode"
    unzip -qq "$DATA_DIR/$DATASET_NAME.zip" "anti-plagiarism-datasets-master/rounds/1314,1315.zip" -d $DATA_DIR/
  else
    unzip "$DATA_DIR/$DATASET_NAME.zip" -d $DATA_DIR
  fi

  mkdir $DATA_PATH

  for round in $(find "$DATA_DIR/anti-plagiarism-datasets-master/rounds" -name "*.zip" -type f)
  do
    unzip "$round" -d "$DATA_DIR/anti-plagiarism-datasets-master/rounds"
    round_dir="${round%.zip}"
    find "$round_dir"/*  -type d -name "*[A-D]" -exec mv {} "$DATA_PATH" \;
    rm -rf "$round_dir"
    rm "$round"
  done
  rm -rf $DATA_DIR/anti-plagiarism-datasets-master

  # Splitting dataset on train/test/val parts
  echo "Splitting on train/test/val"
  sh "$SPLIT_SCRIPT" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART"
  rm -rf "$DATA_PATH"
  mv "$DATA_PATH"_split "$DATA_PATH"
fi
echo "Extracting paths using astminer. You need to specify the path to .jar in \"ASTMINER_PATH\" variable first"
if [ -d "$DATA_PATH"_parsed ]
then
  rm -rf "$DATA_PATH"_parsed
fi
mkdir "$DATA_PATH"_parsed
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang cpp --project "$DATA_PATH"/train --output "$DATA_PATH"_parsed/train --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang cpp --project "$DATA_PATH"/test --output "$DATA_PATH"_parsed/test --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang cpp --project "$DATA_PATH"/val --output "$DATA_PATH"_parsed/val --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
for folder in $(find "$DATA_PATH"_parsed/*/cpp -type d)
do
  for file in "$folder"/*
  do
    type="$(basename -s .csv "$(dirname "$folder")")"
    mv "$file" "$DATA_PATH"_parsed/"$(basename "${file%.csv}.$type.csv")"
  done
  rm -rf "$(dirname "$folder")"
done
