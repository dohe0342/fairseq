mode="w2v" 
instance="p4"

if [ $mode == "w2v" ]
then
: << "END"
	fairseq-hydra-train \
			--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
			--config-name vox_960h_aws_$instance \
			task.data=/opt/ml/code/LibriSpeech/manifests \
			model.w2v_path=/opt/ml/input/data/model/wav2vec_vox_new.pt \
			checkpoint.save_dir=/opt/ml/model 
	rm /opt/ml/model/crash.pt
END
	for i in {0..29}
		do
			init=$(($i % 4))
			if [ $init -eq 0 ]
			then
				fairseq-hydra-train \
					--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
					--config-name base_960h_aws_$instance \
					task.data=/opt/ml/code/LibriSpeech/manifests \
					task.normalize=false \
					model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
					checkpoint.save_dir=/opt/ml/model \
					criterion._name=viewmaker \
					+model.viewmaker=true \
					+model.init_viewmaker=true

			else
				fairseq-hydra-train \
					--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
					--config-name base_960h_aws_$instance \
					task.data=/opt/ml/code/LibriSpeech/manifests \
					task.normalize=false \
					model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
					checkpoint.save_dir=/opt/ml/model \
					criterion._name=viewmaker \
					+model.viewmaker=true
			fi
		done
	cp -r cd /opt/ml/code/fairseq/outputs /opt/ml/model/
	rm /opt/ml/model/crash.pt
elif [ $mode == "hubert" ]
then
	#for i in {0..9}
	#do
	#done
	fairseq-hydra-train \
		--config-dir /opt/ml/code/fairseq/examples/hubert/config/finetune \
		--config-name base_100h_aws_$instance \
		task.label_dir=/opt/ml/code/LibriSpeech/manifests \
		task.data=/opt/ml/code/LibriSpeech/manifests \
		task.normalize=false \
		model.w2v_path=/opt/ml/input/data/model/hubert_base_ls960.pt \
		criterion._name=viewmaker \
		checkpoint.save_dir=/opt/ml/model \
		+model.viewmaker=true
	rm /opt/ml/model/crash.pt

elif [ $mode == "wavlm" ]
then
    for i in {0..0}
    do  
        fairseq-hydra-train \
            --config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
            --config-name base_100h_whale \
            task.data=/opt/ml/code/LibriSpeech/manifests \
            model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
            checkpoint.save_dir=/opt/ml/model \
            +model.wavlm=true
            #criterion._name=viewmaker \
            #+model.viewmaker=true \
            #+model.init_viewmaker=true
    done

else
	for i in {0..9}
	do
		fairseq-hydra-train \
			--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
			--config-name base_100h_aws_$instance \
			common.user_dir=examples/data2vec \
			task.data=/opt/ml/code/LibriSpeech/manifests \
			task.normalize=true \
			model.w2v_path=/opt/ml/input/data/model/audio_base_ls.pt \
			criterion._name=viewmaker \
			checkpoint.save_dir=/opt/ml/model \
			+model.viewmaker=true
		rm /opt/ml/model/crash.pt
	done
fi

#for i in {0..9}
	#do
	#	fairseq-hydra-train \
	#		--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
	#		--config-name base_100h_aws_$instance \
	#		common.user_dir=examples/data2vec \
	#		task.data=/opt/ml/code/LibriSpeech/manifests \
	#		task.normalize=false \
	#		model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
	#		criterion._name=viewmaker \
	#		checkpoint.save_dir=/opt/ml/model \
	#		+model.viewmaker=true
	#	rm /opt/ml/model/crash.pt
	#done


