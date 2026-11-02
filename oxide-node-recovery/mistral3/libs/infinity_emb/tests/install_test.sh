#!/bin/bash

counter=0

while [ $counter -lt 50 ]
do
    # Your command goes here
    # For example, let's echo the current count
    echo "Loop iteration: $counter"
    
    
    pip install infinity-emb==0.0.5 --no-cache-dir

    # Sleep for a certain amount of time 
    random_sleep=$(( ($counter % 60) + 300 ))  

    # Sleep for the random amount of time
    # sleep $random_sleep

    pip uninstall infinity-emb -y

    sleep 1

    # Increment the counter
    counter=$((counter+1))
done
