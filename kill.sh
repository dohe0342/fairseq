kill -9 $(ps -ef | grep [python3] | grep -v grep | awk '{print $2}')
