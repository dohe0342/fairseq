import os 
import time 

dirToWatchList = [
            "./",
            "./workspace",
            "./scripts/", 
            "./scripts/whale/", 
            "./scripts/summit/", 
            "./scripts/aws/", 
            "./scripts/ant/", 
            "./fairseq_cli/", 
            "./fairseq/", 
            "./fairseq/tasks/", 
            "./fairseq/optim/",
            "./fairseq/criterions/",
            "./fairseq/modules/", 
            "./fairseq/models/", 
            "./fairseq/models/wav2vec/",
            "./fairseq/models/roberta/",
            "./fairseq/models/transformer/", 
            "./examples/wav2vec/",
            "./examples/wav2vec/config/",
            "./examples/wav2vec/config/finetuning/",
            "./examples/wav2vec/config/pretraining/",
            "./examples/data2vec/",
            "./examples/data2vec/models/",
            "./examples/data2vec/config/audio/pretraining/",
            "./examples/speech_recognition/",
            "./examples/speech_recognition/new/",
            "./examples/speech_recognition/new/conf/",
            "./examples/speech_recognition/new/decoders/",
            "./examples/roberta/config/pretraining/",
            "./examples/roberta/config/finetuning/"
            ]

lastmod = [int(os.path.getmtime(dirToWatch)) for dirToWatch in dirToWatchList]
#lastmod = int(os.path.getmtime(dirToWatch))

while True:
    for enum, dirToWatch in enumerate(dirToWatchList):
        if lastmod[enum] != int(os.path.getmtime(dirToWatch)): 
            #print('Warning: Modify Detected.') 
            os.system('./git.sh')
            lastmod[enum] = int(os.path.getmtime(dirToWatch)) 
    time.sleep(0.5)
