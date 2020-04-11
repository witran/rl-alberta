# remove weird file prefix after downloading from coursera notebook env
for f in utf-8\'\'*; do mv "$f" "$(echo "$f" | sed s/utf-8\'\'//)"; done
