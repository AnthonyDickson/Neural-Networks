find results -maxdepth 1 -type d | wc -l | (read n; echo Items: $n) 
du -h -d 0 results
