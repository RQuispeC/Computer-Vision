#!/bin/bash

ziped_dir="p3-192617-192618"

rm -rf $ziped_dir
rm -f "$ziped_dir.zip"

mkdir $ziped_dir
mkdir "$ziped_dir/src"
cp -R src/*.py "$ziped_dir/src"
mkdir "$ziped_dir/output"
mkdir "$ziped_dir/input"
cp -R my-output "$ziped_dir"
cp Makefile $ziped_dir
cp report/report.pdf $ziped_dir
zip -r "$ziped_dir.zip" $ziped_dir
rm -rf $ziped_dir



