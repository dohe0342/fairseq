python fairseq_cli/hydra_train.py -m --config-dir /home/work/workspace/fairseq/examples/data2vec/config/audio/pretraining \
  --config-name base_librispeech \
  task.data=/home/work/workspace/LibriSpeech/manifests common.user_dir=examples/data2vec #model._name=$1

#python ~/workspace/send_email.py -c "data2vec kd done (error maybe) :( "
