
""" Masked GPT2 model configuration. It replicates the class `~transformers.BertConfig`
and adapts it to the specificities of MaskedBert (`pruning_method`, `mask_init` and `mask_scale`."""


import logging
from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)


class MaskedGPT2Config(PretrainedConfig):
    """
    A class replicating the `~transformers.gpt2-medium config` with additional parameters for pruning/masking configuration.
    """

    model_type = "masked_gpt2"

    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            bos_token_id=50256,
            eos_token_id=50256,
            pruning_method="topK",
            mask_init="constant",
            mask_scale=0.0,
            **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # pruning
        self.pruning_method = pruning_method
        self.mask_init = mask_init
        self.mask_scale = mask_scale
