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

lastmod = [int(os.path.getmtime(dirToWatch)) for dirToWatch in dirToWatchList]
#lastmod = int(os.path.getmtime(dirToWatch))

while True:
    for enum, dirToWatch in enumerate(dirToWatchList):
        if lastmod[enum] != int(os.path.getmtime(dirToWatch)): 
            #print('Warning: Modify Detected.') 
            os.system('./git.sh')
            lastmod[enum] = int(os.path.getmtime(dirToWatch)) 
    time.sleep(0.3)
