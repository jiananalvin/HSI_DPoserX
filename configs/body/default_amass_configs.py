from posix import truncate
from configs.general_configs import get_general_configs
import ml_collections


def get_default_configs():
    config = get_general_configs()
    # original
    config.devices = [0]
    config.name = 'default'
    config.dataset = 'amass'

    # data
    data = config.data
    data.normalize = True
    data.rot_rep = 'axis'  # rot6d or axis
    data.min_max = False  # Z-score or min-max Normalize
    data.include_global_orient = True  # Include global orientation (root rotation) in pose representation

    # training
    training = config.training
    config.training.batch_size = 64  # 1280
    training.n_iters = 20000  # 200000
    training.log_freq = 100   # Log every 100 iterations 
    training.eval_freq = 300  # 10000
    training.save_freq = 450  # 15000
    training.auxiliary_loss = True  # not recommended
    training.denoise_steps = 10  # for computing auxiliary loss
    training.render = True  # render results while validating
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True
    # Ambient Diffusion, not used
    training.random_mask = False
    training.min_mask_rate = 0.0
    training.max_mask_rate = 0.2
    training.observation_type = 'zeros'

    # evaluation
    evaluate = config.eval
    evaluate.batch_size = 32  # 50

    # optimization
    optim = config.optim
    optim.optimizer = 'RAdam'
    optim.lr = 2e-4
    optim.weight_decay = 0.0
    optim.warmup = 150  # 5000

    # Text encoder configuration (default for all body configs, can be overridden in specific configs)
    # Options:
    #   - 'clip': Original CLIP (77 tokens max) - default for backward compatibility
    #   - 'sentence-bert': Sentence-BERT (~512 tokens) - recommended for long text
    #   - 't5': T5 encoder (~512 tokens) - good for understanding
    #   - 'longformer': Longformer (~4096 tokens) - for very long documents
    config.text_encoder = text_encoder = ml_collections.ConfigDict()
    text_encoder.type = 'sentence-bert'  # Default to CLIP for backward compatibility
    text_encoder.model_name = None  # Uses default for each type, or specify: 'all-mpnet-base-v2', 't5-base', etc.

    config.seed = 42

    return config
