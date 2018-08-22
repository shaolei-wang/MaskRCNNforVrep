#!/bin/bash
ls *.jpg |awk -F "rgb_" '{print "mv "$0" "$1$2""}'|bash
