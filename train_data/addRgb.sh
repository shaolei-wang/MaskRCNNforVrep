#!/bin/sh
for files in $(ls *.png)
    do mv $files "rgb_"$files
done
