git pull
mode="w2v"
exp_name=$1

if [ $mode == "w2v" ]
then
	for i in {0..29}
	do
		init=$(($i % 4)) 
		if [ $init -eq 2 ] 
		then
			fairseq-hydra-train \
				--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
				--config-name base_100h_whale \
				task.data=/home/work/workspace/LibriSpeech/manifests \
				model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_vox_new.pt \
				checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
				criterion._name=viewmaker \
				+model.viewmaker=true \
				+model.init_viewmaker=true 
		else
			fairseq-hydra-train \
				--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
				--config-name vox_960h_whale \
				task.data=/home/work/workspace/LibriSpeech/manifests \
				model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_vox_new.pt \
				checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
				criterion._name=viewmaker \
				+model.viewmaker=true 
		fi
	done

elif [ $mode == "hubert" ]
then
	for i in {0..29}
	do
		fairseq-hydra-train \
			--config-dir /home/work/workspace/fairseq/examples/hubert/config/finetune \
			--config-name base_100h_whale \
			task.label_dir=/home/work/workspace/LibriSpeech/manifests \
			task.data=/home/work/workspace/LibriSpeech/manifests \
			task.normalize=false \
			model.w2v_path=/home/work/workspace/models/hubert_model/hubert_base_ls960.pt \
			criterion._name=viewmaker \
			checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
			+model.viewmaker=true
	done

elif [ $mode == "wavlm" ]
then
	for i in {0..29}
	do
		fairseq-hydra-train \
			--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
			--config-name base_100h_whale \
			task.data=/home/work/workspace/LibriSpeech/manifests \
			model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt \
			checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
			+model.wavlm=true \
			criterion._name=viewmaker \
			+model.viewmaker=true 
			#+model.init_viewmaker=true
	done

else
	for i in {0..29}
	do
		fairseq-hydra-train \
			--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
			--config-name base_100h_whale \
			common.user_dir=examples/data2vec \
			task.data=/home/work/workspace/LibriSpeech/manifests \
			task.normalize=true \
			model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
			criterion._name=viewmaker \
			checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
			+model.viewmaker=true
	done
fi

:<<'END'
for i in {0..0} ; do
    if [ $i -eq 0 ] ; then
    mkdir /home/work/workspace/fairseq/scripts/whale/outputs/$1
    #cp /home/work/workspace/fairseq/scripts/whale/outputs/pretrained_lightweight_viewmaker.pt /home/work/workspace/fairseq/scripts/whale/outputs/$1/checkpoint_last.pt
    fi  
    fairseq-hydra-train \
        --config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
        --config-name base_100h_whale \
        common.user_dir=examples/data2vec \
        task.data=/home/work/workspace/LibriSpeech/manifests \
        task.normalize=true \
        model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
        criterion._name=viewmaker \
        checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
        +model.viewmaker=true \
done

fairseq-hydra-train \
		--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h_ant \
		common.user_dir=examples/hubert \
		task.data=/workspace/LibriSpeech/manifests \
		task.normalize=false \
		+task.fine_tuning=true \
		task.labels=["ltr"] \
		+task.single_target=true \
		model._name=hubert_ctc \
		model.w2v_path=/workspace/models/hubert_model/hubert_base_ls960.pt \
		criterion._name=viewmaker \
		checkpoint.save_dir=/workspace/fairseq/scripts/whale/outputs/$1 \
		+model.viewmaker=true

END
