from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder
from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2CtcConfig

class Wav2Vec2Ctc(nn.Module):
    def __init__(
            self,
    ):
        super(RepresenationModel, self).__init__()
        
        #w2v_path = self.download_pretrained_model()
        w2v_path = '/home/work/workspace/models/wav2vec_model/wav2vec_small.pt'
        cfg = Wav2Vec2CtcConfig()
        cfg.w2v_path = w2v_path
        cfg.normalize = False
        cfg._name = "wav2vec"

        self.normalize = cfg.normalize
        self.w2v_encoder = Wav2VecEncoder(cfg, 32)

        function_type = type(self.w2v_encoder.w2v_model._get_feat_extract_output_lengths)
        self.w2v_encoder.w2v_model._get_feat_extract_output_lengths = function_type(
            self._get_feat_extract_output_lengths_warp, self.w2v_encoder.w2v_model
            )

        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    #FIXME: More clever way!
    @staticmethod
    def _get_feat_extract_output_lengths_warp(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() // stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def download_pretrained_model(self):
        if self.config.model == 'k-w2v':
            path = "/models/checkpoint_best_new.pt"

        elif self.config.model == 'xlsr':
            old_path = "/models/xlsr2_300m.pt"
            path = "/models/xlsr2_300m_new.pt"

            model = torch.load(old_path)
            w2v_b = torch.load("/models/checkpoint_best_new.pt")
            del model['cfg']['task']
            model['cfg']['task'] = w2v_b['cfg']['task']
            model['cfg']['task']['normalize'] = True
            torch.save(model, path)
            del model
            del w2v_b

        return path

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        if self.normalize:
            with torch.no_grad():
                inputs = F.layer_norm(inputs, inputs.shape)
        masks = lengths_to_padding_mask(input_lengths)

        inputs = inputs.unsqueeze(0) if len(inputs.size()) == 1 else inputs
        res = self.w2v_encoder(inputs, masks)

        outputs = res["encoder_out"]
        outputs = outputs.permute(1, 0, 2)
        bs = outputs.size(0)
        masks = res["padding_mask"]
        if masks is not None:
            output_lengths = (~masks).sum(dim=1)  # (B)
            output_lengths = output_lengths.tolist()
        else:
            output_lengths = torch.IntTensor(
                [outputs.shape[1]]
            ).repeat(bs).to(outputs.device).tolist()

        outputs = outputs.log_softmax(dim=-1)

        return outputs, output_lengths
