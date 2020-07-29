#!/bin/bash
# Run script from code2seq dir using command:
#    sh scripts/download_data.sh
TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
SHUFFLE=$5
DATA_DIR=./data
DATASET_NAME=cf
ASTMINER_PATH=../astminer/build/shadow/lib-0.5.jar
SPLIT_SCRIPT=./scripts/split_dataset.sh

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
  if [ ! -f "$DATA_DIR/cf.zip" ]
  then
    aws s3 cp s3://datasets.ml.labs.aws.intellij.net/codeforces-code-clone/anti-plagiarism-datasets-master.zip "$DATA_DIR/cf.zip"
  fi

  echo "Unzip dataset"

  if [ $DEV ]
  then
    echo "Dev mode"
    unzip -qq "$DATA_DIR/cf.zip" "anti-plagiarism-datasets-master/rounds/1314,1315.zip" -d $DATA_DIR/
  else
    unzip "$DATA_DIR/cf.zip" -d $DATA_DIR/
  fi
  mv "$DATA_DIR"/anti-plagiarism-datasets-master/rounds "$DATA_PATH"
  rm -rf "$DATA_DIR"/anti-plagiarism-datasets-master

  for file in $(find "$DATA_PATH" -type f)
  do
    unzip "$file" -d "$DATA_PATH"
    competition="${file%.zip}"
    for task in $(find "$competition"/* -type d)
    do
      echo "Moving $task to $DATA_PATH"
      mv "$task" "$DATA_PATH"
    done
    echo "Remove $competition"
    rm -rf "$competition"
    rm "$file"
  done
  ls "$DATA_PATH"
  if [ $DEV ]
  then
    echo "Dev mode"
    find "$DATA_PATH"/* -type d -name "*E" -exec rm -rf {} \;
    find "$DATA_PATH"/* -type d -name "*F" -exec rm -rf {} \;
    find "$DATA_PATH"/* -type d -name "*G" -exec rm -rf {} \;
    find "$DATA_PATH"/* -type d -name "*H" -exec rm -rf {} \;
  fi

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
java -jar -Xmx4096m $ASTMINER_PATH code2vec --lang cpp --project "$DATA_PATH"/train --output "$DATA_PATH"_parsed/train --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx4096m $ASTMINER_PATH code2vec --lang cpp --project "$DATA_PATH"/test --output "$DATA_PATH"_parsed/test --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx4096m $ASTMINER_PATH code2vec --lang cpp --project "$DATA_PATH"/val --output "$DATA_PATH"_parsed/val --maxH 8 --maxW 2 --granularity file --folder-label --split-tokens
for folder in $(find "$DATA_PATH"_parsed/*/cpp -type d)
do
  for file in "$folder"/*
  do
    mv "$file" "$DATA_PATH"_parsed/"$(basename "${file/.csv/.$(basename "$(dirname "$folder")").csv}")"
  done
  rm -rf "$(dirname "$folder")"
done