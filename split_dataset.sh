#!/bin/bash
# Script - split data between train and test
# Default values
train=70
val=10
test=20
original_dataset_path="./data/poj_104"
split_dataset_path="./data/poj_104_split"
dir_train=${split_dataset_path}"/train"
dir_validation=${split_dataset_path}"/validation"
dir_test=${split_dataset_path}"/test"

echo "Train $train % "
echo "Val $val %"
echo "Test $test %"
echo "Original dataset path: ${original_dataset_path}"
echo "Train dataset path: ${dir_train}"
echo "Val dataset path = ${dir_validation}"
echo "Test dataset path = ${dir_test}"

echo ""
echo "Removing all data inside "${split_dataset_path}
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
