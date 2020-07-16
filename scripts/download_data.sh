#!/bin/bash
# Run script from code2seq dir using command:
#    sh scripts/download_data.sh
TRAIN_SPLIT_PART=60
VAL_SPLIT_PART=20
TEST_SPLIT_PART=20
DEV=false
DATA_DIR=./data
ASTMINER_PATH=../astminer/build/shadow/lib-0.5.jar
SPLIT_SCRIPT=./scripts/split_dataset.sh

function is_int(){
  if [[ ! "$1" =~ ^[+-]?[0-9]+$ ]]; then
    echo "Non integer {$1} passed in --$2-part"
    exit 1
  fi
}

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset=NAME             specify dataset name, available: java-small, java-med, java-large, poj_104"
      echo "--train-part=VAL               specify a percentage of dataset used as train set"
      echo "--test-part=VAL                specify a percentage of dataset used as test set"
      echo "--val-part=VAL                 specify a percentage of dataset used as validation set"
      exit 0
      ;;
    -d|--dataset*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        DATASET_NAME=$2
        shift 2
      else
        echo "Specify dataset name"
        exit 1
      fi
      ;;
    --train-part*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        is_int "$2" "train"
        TRAIN_SPLIT_PART=$2
        shift 2
      else
        echo "Specify train part"
        exit 1
      fi
      ;;
    --test-part*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        is_int "$2" "test"
        TEST_SPLIT_PART=$2
        shift 2
      else
        echo "Specify test part"
        exit 1
      fi
      ;;
    --val-part*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        is_int "$2" "val"
        VAL_SPLIT_PART=$2
        shift 2
      else
        echo "Specify val part"
        exit 1
      fi
      ;;
    --dev*)
      shift
      DEV=true
      shift
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

function load_java_dataset(){
  if [ ! -d "$DATA_PATH" ]
  then
    if [ ! -f "$DATA_DIR"/"$1"-preprocessed.tar.gz ]
    then
      echo "Downloading dataset $1"
      wget https://s3.amazonaws.com/code2seq/datasets/$1-preprocessed.tar.gz -P $DATA_DIR/
    else
      echo "Dataset $1 already downloaded"
    fi
    echo "Unzip dataset"
    tar -xvzf $DATA_DIR/$1-preprocessed.tar.gz -C data/
  else
    echo "Dataset $1 already exists"
  fi
}

if [[ "$DATASET_NAME" == "java-"* ]]
then
  load_java_dataset "$DATASET_NAME"
elif [ "$DATASET_NAME" = "poj_104" ]
then
  echo "Downloading dataset $1"
  if [ -d "$DATA_PATH" ]
  then
    echo "$DATA_PATH exists."
  else
    if [ ! -f "$DATA_DIR/poj-104-original.tar.gz" ]
    then
      wget https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz -P $DATA_DIR/
    fi

    echo "Unzip dataset"
    tar -xvzf "$DATA_DIR/poj-104-original.tar.gz" -C $DATA_DIR/
    mv ./"$DATA_DIR"/ProgramData ./"$DATA_PATH"

    # In the developer mode we leave only several classes
    if [ $DEV ]
    then
      echo "Dev mode"
      find "$DATA_PATH"/* -type d -name "1[0-9][0-9]" -exec rm -rf {} \;
      find "$DATA_PATH"/* -type d -name "[1-9][0-9]" -exec rm -rf {} \;
    fi

    # To prepare our dataset for astminer we need to rename all .txt files to .c files
    echo "Renaming files"
    find "$DATA_PATH"/*  -name "*.txt" -type f -exec sh -c 'mv "$0" "${0%.txt}.c"' {} \;
    echo "Splitting on train/test/val"
    # Splitting dataset on train/test/val parts
    sh "$SPLIT_SCRIPT" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART" --shuffle
    rm -rf "$DATA_PATH"
    mv "$DATA_PATH"_split "$DATA_PATH"
  fi
  echo "Extracting AST using astminer. You need to clone astminer first"
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
else
  echo "Dataset $DATASET_NAME does not exist"
fi
