#!/bin/bash

ziped_dir="p2-192617-192618"

rm -rf $ziped_dir
rm -f "$ziped_dir.zip"

mkdir $ziped_dir
mkdir "$ziped_dir/src"
cp -R src/*.py "$ziped_dir/src"
cp -R input $ziped_dir
mkdir "$ziped_dir/output"
cp Makefile $ziped_dir
cp report/report.pdf $ziped_dir
zip -r "$ziped_dir.zip" $ziped_dir
rm -rf $ziped_dir



