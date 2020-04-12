#!/bin/sh
# remove weird file prefix after downloading from coursera notebook env
for f in utf-8\'\'*; do 
  if [ ! -e "$f" ]; then break; fi
  mv "$f" "$(echo "$f" | sed s/utf-8\'\'//)"; 
done
