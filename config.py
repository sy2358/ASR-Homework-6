class config():
    dim = 123
    phngroup_filename = "phngroup"
    max_iter = None
    nepochs = 20
    batch_size = 20
    # dropout = 0.2
    lr = 0.001
    lr_decay = 0.9
    nepoch_no_imprv = 3
    hidden_size = 300
    output_path = "results/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    nb_layers = 2
    keep_prob = 0.5
