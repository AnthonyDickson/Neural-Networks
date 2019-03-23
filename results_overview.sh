find results -type d -links 2 | wc -l | (read n; echo Items: $n) 
du results -h -d 0 
