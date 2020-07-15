#!/bin/bash
# Script - split data between train and test
# Default values
train=60
val=20
test=20
original_dataset_path="./data/poj_104"
split_dataset_path="./data/poj_104_split"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-i, --input-dir=DIR       specify a directory where dataset is located"
      echo "-o, --output-dir=DIR      specify a directory to store output in"
      echo "--train=VAL               specify a percentage of dataset used as train set"
      echo "--test=VAL                specify a percentage of dataset used as test set"
      echo "--val=VAL                 specify a percentage of dataset used as validation set"
      exit 0
      ;;
    -i|--input-dir*)
      shift
      if test $# -gt 0; then
        original_dataset_path=$1
      else
        echo "no input dir specified, using default: ./data/poj_104"
        exit 1
      fi
      shift
      ;;
    -o|--output-dir*)
      shift
      if test $# -gt 0; then
        split_dataset_path=$1
      else
        echo "no output dir specified, using default: ./data/poj_104_split"
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
    *)
      echo "something went wrong"
      exit 1
  esac
done

dir_train="${split_dataset_path}/train"
dir_validation="${split_dataset_path}/val"
dir_test="${split_dataset_path}/test"

echo "Train $train % "
echo "Val $val %"
echo "Test $test %"
echo "Original dataset path: ${original_dataset_path}"
echo "Train dataset path: ${dir_train}"
echo "Val dataset path = ${dir_validation}"
echo "Test dataset path = ${dir_test}"

echo ""
echo "Removing all data inside ${split_dataset_path}"
rm -rf $split_dataset_path
mkdir $split_dataset_path

mkdir $dir_train
mkdir $dir_validation
mkdir $dir_test

cp -r $original_dataset_path/* $dir_train/

find $dir_train/* -type d -exec basename {} \; | while read dir_class
do
    echo "Splitting class - $dir_class";
    mkdir "$dir_validation/$dir_class"
    mkdir "$dir_test/$dir_class"
    num_files=$(find "$dir_train/$dir_class" -type f | wc -l)

    train_bound=$(python -c "print(int($num_files * $train / 100))")
    test_bound=$(python -c "print(int($train_bound + $num_files * $test / 100))")

    counter=0
    files=$(find "$dir_train/$dir_class" -type f -exec basename {} \;| sort -R)
    for file in $files;
    do
        (( counter += 1 ))
        if [[ $counter -gt train_bound && $counter -le test_bound ]]; then
            mv "$dir_train/$dir_class/$file" "$dir_validation/$dir_class/$file"
        fi
        if [[ $counter -gt test_bound ]]; then
            mv "$dir_train/$dir_class/$file" "$dir_test/$dir_class/$file"
        fi
    done
done

echo "Done"