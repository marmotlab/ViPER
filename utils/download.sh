#!/bin/bash

MAP_TEST_URL="https://bit.ly/viper_test_map"
MAP_TRAIN_URL="https://bit.ly/viper_train_map"
MODEL_URL="https://www.dropbox.com/scl/fi/fg3vody0t1mhdkan6o7e4/model.zip?rlkey=bkayp9q26zkkkfg7ttxeblg9d&st=jnt8gjwu&dl=1"
MAP_TEST="maps_test.zip"
MAP_TRAIN="maps_train.zip"
MODEL="model.zip"

if command -v wget > /dev/null; then
    echo "Downloading maps_test.zip"
    wget -O "$MAP_TEST" "$MAP_TEST_URL"

    echo "Downloading maps_train.zip"
    wget -O "$MAP_TRAIN" "$MAP_TRAIN_URL"

    echo "Downloading ViPER model"
    wget -O "$MODEL" "$MODEL_URL"
else
    echo "Error: wget is not installed."
    exit 1
fi


if command -v unzip > /dev/null; then
    unzip "$MAP_TEST"
    unzip "$MAP_TRAIN"
    unzip "$MODEL"
    rm "$MAP_TEST"
    rm "$MAP_TRAIN"
    rm "$MODEL"
else
    echo "Error: unzip is not installed."
    exit 1
fi



