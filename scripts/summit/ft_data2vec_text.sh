git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name mnli \
	task.data=/workspace/fairseq/examples/roberta/MNLI-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt \
	criterion._name=sentence_prediction_viewmaker \
	optimization.max_update=100000 \
	optimization.max_epoch=100 \
	dataset.batch_size=16
	#+model.branch_ctc_v1=false \
	#+model.branch_ctc_v2=false \
	#+model.branch_ctc_v3=false 
