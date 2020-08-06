#!/bin/bash
# Run script from code2seq dir using command:
#    sh scripts/download_data.sh
TRAIN_SPLIT_PART=60
VAL_SPLIT_PART=20
TEST_SPLIT_PART=20
DEV=false
LOAD_SPLITTED=false
DATA_DIR=./data
POJ_DOWNLOAD_SCRIPT=./scripts/download_poj.sh
CODEFORCES_DOWNLOAD_SCRIPT=./scripts/download_codeforces.sh
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
      echo "--dev                          pass it if developer mode should be used, default false"
      echo "--load-splitted                pass it if splitted dataset needs to be loaded, available only for poj_104, default false"
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
      DEV=true
      shift
      ;;
    --load-splitted*)
      LOAD_SPLITTED=true
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

if [[ "$DATASET_NAME" == "java-"* ]]
then
  if [ ! -d "$DATA_PATH" ]
  then
    if [ ! -f "$DATA_DIR"/"$DATASET_NAME"-preprocessed.tar.gz ]
    then
      echo "Downloading dataset $DATASET_NAME"
      wget https://s3.amazonaws.com/code2seq/datasets/"$DATASET_NAME"-preprocessed.tar.gz -P $DATA_DIR/
    else
      echo "Dataset $DATASET_NAME already downloaded"
    fi
    echo "Unzip dataset"
    tar -xvzf $DATA_DIR/"$DATASET_NAME"-preprocessed.tar.gz -C data/
  else
    echo "Dataset $DATASET_NAME already exists"
  fi
elif [ "$DATASET_NAME" == "poj_104" ]
then
  sh "$POJ_DOWNLOAD_SCRIPT" "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART" "$DEV" "$ASTMINER_PATH" "$SPLIT_SCRIPT" "$LOAD_SPLITTED"
elif [ "$DATASET_NAME" == "codeforces" ]
then
  sh "$CODEFORCES_DOWNLOAD_SCRIPT" "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART" "$DEV" "$ASTMINER_PATH" "$SPLIT_SCRIPT"
else
  echo "Dataset $DATASET_NAME does not exist"
fi