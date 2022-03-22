git pull
source ~/.bashrc
model=$1

echo "====================   $model // $subset   ===================="
CUDA_VISIBLE_DEVICES=$2 python /home/work/workspace/fairseq/examples/speech_recognition/new/train_spk_clf_conv.py \
	--config-dir /home/work/workspace/fairseq/examples/speech_recognition/new/conf \
	--config-name infer \
	task=audio_finetuning \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	common.user_dir=examples/data2vec \
	task.labels=ltr \
	decoding.type=viterbi \
	decoding.unique_wer_file=False \
	dataset.gen_subset=train-100 \
	common_eval.path=/home/work/workspace/models/data2vec_model/$model \
	distributed_training.distributed_world_size=1
