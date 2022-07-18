fairseq-hydra-train \
    --config-dir /opt/ml/code/fairseq/examples/hubert/config/finetune \
    --config-name base_100h_ant \
    task.label_dir=/opt/ml/code/LibriSpeech/manifests \
    task.data=/opt/ml/code/LibriSpeech/manifests \
    task.normalize=false \
    model.w2v_path=/opt/ml/input/data/model/hubert_base_ls960.pt \
    criterion._name=ctc \
    checkpoint.save_dir=/opt/ml/model 
