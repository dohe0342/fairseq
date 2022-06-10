git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name rte \
	task.data=/workspace/fairseq/examples/roberta/RTE-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt \
	criterion._name=sentence_prediction_viewmaker \
	optimization.max_update=6108 \
	optimization.max_epoch=30
	#+model.branch_ctc_v1=false \
	#+model.branch_ctc_v2=false \
	#+model.branch_ctc_v3=false 
