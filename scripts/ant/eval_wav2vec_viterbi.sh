subset=$1
model=$2
CUDA_VISIBLE_DEVICES=$3 python /workspace/fairseq/examples/speech_recognition/infer.py \
  /workspace/LibriSpeech/manifests \
  --task audio_finetuning \
  --nbest 1 \
  --path /workspace/models/$model \
  --gen-subset $subset \
  --results-path /workspace/fairseq/examples/wav2vec/ctc_eval \
  --w2l-decoder fairseqlm \
  --lexicon /workspace/models/lm_model/librispeech_lexicon.lst \
  --lm-model /workspace/models/lm_model/lm_librispeech_word_transformer.pt \
  --lm-weight 0.87 \
  --word-score -1 \
  --sil-weight 0 \
  --criterion ctc \
  --labels ltr \
  --max-tokens 4000000 \
  --post-process letter \
  --beam=1
