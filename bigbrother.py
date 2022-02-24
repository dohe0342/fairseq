import os
import time

dirToWatchList = [
            "./",
            "./scripts/", 
            "./scripts/whale/", 
            "./scripts/summit/", 
            "./fairseq/", 
            "./fairseq/tasks/", 
            "./fairseq/optim/",
            "./fairseq/criterions/",
            "./fairseq/models/", 
            "./fairseq/models/wav2vec/",
            "./examples/wav2vec/",
            "./examples/data2vec/"
            ]

fileList = []
timeToSleep = 0.3
count = 0

while True:
    for enum, dirToWatch in enumerate(dirToWatchList):
        newFileList = os.listdir(dirToWatch)
        if count==0: fileList.append(newFileList)         
        if fileList[enum] != newFileList:
            print(f"changes detected in {dirToWatch}:")
            changes = set(newFileList) - set(fileList[enum])
            print(changes)
            fileList[enum] = newFileList
        else:
            print("no new changes")
    count += 1
    time.sleep(timeToSleep)
