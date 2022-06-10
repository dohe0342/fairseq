fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name rte \
	task.data=/workspace/fairseq/examples/roberta/RTE-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt \
	+model.branch_ctc_v1=false \
	+model.branch_ctc_v2=false \
	+model.branch_ctc_v3=false 
