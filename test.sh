#!/bin/bash

for i in {1..150}
do
    echo $i
    python maxpp_paperversion2.py >> test1.txt
done
