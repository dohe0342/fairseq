import os 
import time 
dirToWatchList = [
            "./",
            "./workspace",
            "./scripts/", 
            "./scripts/whale/", 
            "./scripts/summit/", 
            "./scripts/ant/", 
            "./scripts/aws/", 
            "./fairseq_cli/", 
            "./fairseq/", 
            "./fairseq/tasks/", 
            "./fairseq/optim/",
            "./fairseq/criterions/",
            "./fairseq/modules/", 
            "./fairseq/models/", 
            "./fairseq/models/wav2vec/",
            "./fairseq/models/hubert/",
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
            "./examples/roberta/config/finetuning/",
            "./examples/hubert/",
            "./examples/hubert/config/",
            "./examples/hubert/config/finetune/",
            ]

#lastmod = [dirpath[0] for dirpath in os.walk(path)]
#lastmod = [int(os.path.getmtime(dirToWatch)) for dirToWatch in dirToWatchList]
lastmod = []
dirToWatchList = []
path = './'
for dirpath in os.walk(path):
    if '.git' in dirpath[0] or 'lib' in dirpath[0] or 'bin' in dirpath[0] or 'build' in dirpath[0]:
        continue
    dirToWatch = dirpath[0]+'/'
    lastmod.append(int(os.path.getmtime(dirToWatch)))
    dirToWatchList.append(dirToWatch)
#lastmod = int(os.path.getmtime(dirToWatch))

while True:
    for enum, dirToWatch in enumerate(dirToWatchList):
        if lastmod[enum] != int(os.path.getmtime(dirToWatch)): 
            #print('Warning: Modify Detected.') 
            os.system('./git.sh')
            lastmod[enum] = int(os.path.getmtime(dirToWatch)) 
    time.sleep(0.5)
