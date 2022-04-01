import os
import pickle


def qdpy_save_archive(container, play_itr, gen_itr, net_itr, logbook, save_dir):
    with open(os.path.join(save_dir, 'latest-0.p'), 'wb') as f:
        pickle.dump(
            {
                'container': container,
                'net_itr': net_itr,
                'gen_itr': gen_itr,
                'play_itr': play_itr,
                'logbook': logbook,
            }, f)

