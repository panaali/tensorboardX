from numpy.random import rand
from tensorboardX import SummaryWriter

import time

num_layers_options = [2, 3, 4]
hidden_size_options = [128, 256]

for num_layers in num_layers_options:
    for hidden_size in hidden_size_options:
        with SummaryWriter() as writer:
            writer.add_hparams_start(dict(num_layers=num_layers, hidden_size=hidden_size))
            t = rand()
            for n_iter in range(100):
                t += rand() * 0.01
                writer.add_scalar('Valid loss', t + 0.1, n_iter)
                writer.add_scalar('Train Loss', t, n_iter)
            writer.add_hparams_end()
        time.sleep(1)