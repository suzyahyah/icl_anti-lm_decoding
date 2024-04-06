#!/usr/bin/env bash
# Author: Suzanna Sia

start=$1
end=$2

while true; do
    for x in `seq ${start} 1 ${end}`; do
        if [ -e logs_o/*$x ]; then # it's your filen
            echo -n "$x>"; sed -n 1p logs_o/*$x # whatever you want to capture
            #echo $x && cat logs_e/*$x; echo "" 
            cat logs_e/*$x
            echo ""
        fi
    done
    sleep 60
done
