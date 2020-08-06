#!/bin/bash
# Script - split data between train and test
# Default values
# options:
# -h, --help      show brief help
# $1              specify a directory where dataset is located
# $2              specify a directory to store output in
# $3              specify a percentage of dataset used as train set
# $4              specify a percentage of dataset used as test set
# $5              specify a percentage of dataset used as validation set

SHUFFLE=false

ORIGINAL_DATASET_PATH=$1
SPLIT_DATASET_PATH=$2
TRAIN_SPLIT_PART=$3
TEST_SPLIT_PART=$4
VAL_SPLIT_PART=$5

DIR_TRAIN="${SPLIT_DATASET_PATH}/train"
DIR_VAL="${SPLIT_DATASET_PATH}/val"
DIR_TEST="${SPLIT_DATASET_PATH}/test"

echo "Train $TRAIN_SPLIT_PART % "
echo "Val $VAL_SPLIT_PART %"
echo "Test $TEST_SPLIT_PART %"
echo "Shuffle $SHUFFLE"
echo "Original dataset path: ${ORIGINAL_DATASET_PATH}"
echo "Train dataset path: ${DIR_TRAIN}"
echo "Val dataset path = ${DIR_VAL}"
echo "Test dataset path = ${DIR_TEST}"

echo ""
echo "Removing all data inside ${SPLIT_DATASET_PATH}"
rm -rf "$SPLIT_DATASET_PATH"
mkdir "$SPLIT_DATASET_PATH"

mkdir "$DIR_TRAIN"
mkdir "$DIR_VAL"
mkdir "$DIR_TEST"

cp -r "$ORIGINAL_DATASET_PATH"/* "$DIR_TRAIN"/

find "$DIR_TRAIN"/* -type d -exec basename {} \; | while read DIR_CLASS
do
    echo "Splitting class - $DIR_CLASS";
    mkdir "$DIR_VAL/$DIR_CLASS"
    mkdir "$DIR_TEST/$DIR_CLASS"
    num_files=$(find "$DIR_TRAIN/$DIR_CLASS" -type f | wc -l)
    train_bound=$(expr $num_files \* $TRAIN_SPLIT_PART / 100)
    test_bound=$(expr $train_bound + $num_files \* $TEST_SPLIT_PART / 100)

    counter=$(expr 0)

    files=$(find "$DIR_TRAIN/$DIR_CLASS" -type f -exec basename {} \;)

    for file in $files;
    do
        counter=$(expr $counter + 1)
        if [ $counter -gt $train_bound ] && [ $counter -le $test_bound ]; then
            mv "$DIR_TRAIN/$DIR_CLASS/$file" "$DIR_TEST/$DIR_CLASS/$file"
        fi
        if [ $counter -gt $test_bound ]; then
            mv "$DIR_TRAIN/$DIR_CLASS/$file" "$DIR_VAL/$DIR_CLASS/$file"
        fi
    done
done

echo "Done"
