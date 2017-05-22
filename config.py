class config():
    dim = 123
    phngroup_filename = "phngroup"
    max_iter = None
    nepochs = 50
    batch_size = 32
    # dropout = 0.2
    lr = 0.001
    lr_decay = 0.95
    nepoch_no_imprv = 6
    hidden_size = 400
    output_path = "results/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    nb_layers = 1
    keep_prob = 0.5
    variational = True

# From ptb_word_ml.py
# class SmallConfig(object):
#   """Small config."""
#   init_scale = 0.1
#   learning_rate = 1.0
#   max_grad_norm = 5
#   num_layers = 2
#   num_steps = 20
#   hidden_size = 200
#   max_epoch = 4
#   max_max_epoch = 13
#   keep_prob = 1.0
#   lr_decay = 0.5
#   batch_size = 20
#   vocab_size = 10000
#
#
# class MediumConfig(object):jk
#   """Medium config."""
#   init_scale = 0.05
#   learning_rate = 1.0
#   max_grad_norm = 5
#   num_layers = 2
#   num_steps = 35
#   hidden_size = 650
#   max_epoch = 6
#   max_max_epoch = 39
#   keep_prob = 0.5
#   lr_decay = 0.8
#   batch_size = 20
#   vocab_size = 10000
#
#
# class LargeConfig(object):
#   """Large config."""
#   init_scale = 0.04
#   learning_rate = 1.0
#   max_grad_norm = 10
#   num_layers = 2
#   num_steps = 35
#   hidden_size = 1500
#   max_epoch = 14
#   max_max_epoch = 55
#   keep_prob = 0.35
#   lr_decay = 1 / 1.15
#   batch_size = 20
#   vocab_size = 10000