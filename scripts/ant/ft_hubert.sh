fairseq-hydra-train \
    --config-dir /workspace/fairseq/examples/hubert/config/finetune \
    --config-name base_100h_ant \
    model.w2v_path=/workspace/models/hubert_model/hubert_base_ls960.pt \
    task.label_dir=/workspace/LibriSpeech/manifests \
    task.data=/workspace/LibriSpeech/manifests \
    task.normalize=false \
    criterion._name=ctc \
    checkpoint.save_dir=/workspace/fairseq/scripts/ant/$1 
