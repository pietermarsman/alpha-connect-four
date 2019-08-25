import os
import time
from itertools import cycle
from multiprocessing.pool import Pool

from classifier import train_new_model
from game import TwoPlayerGame
from observer import ConsoleObserver, AlphaConnectSerializer
from player import AlphaConnectPlayer
from state import State
from util import list_files, replace_extension


def optimize_once(data_dir, model_path, max_games=None):
    log_path = replace_extension(model_path, '.csv')
    model = train_new_model(data_dir, log_path, max_games)
    model.save(model_path)


def optimize_continuously(model_dir, data_dir, max_games=None, wait=30 * 60):
    os.makedirs(model_dir, exist_ok=True)
    if is_first_model(model_dir):
        _, model_path = new_model_path(model_dir)
        model = train_new_model(None)
        model.save(model_path)

    while True:
        _, model_path = new_model_path(model_dir)
        log_path = replace_extension(model_path, '.csv')
        model = train_new_model(data_dir, log_path, max_games)
        # todo move model after saving to prevent OSError: Unable to open file (unable to lock file, errno = 35,
        #  error message = 'Resource temporarily unavailable')
        model.save(model_path)
        time.sleep(wait)


def simulate_continuously(model_dir, data_dir, processes=1):
    with Pool(processes) as p:
        for _ in p.imap_unordered(simulate_once_with_newest_model, cycle([(model_dir, data_dir)])):
            pass


def simulate_once_with_newest_model(directories):
    model_dir, data_dir = directories
    model_iteration, model_path = latest_model_path(model_dir)
    model_data_dir = os.path.join(data_dir, '%6.6d' % model_iteration)
    simulate_once(model_path, model_data_dir)


def simulate_once(model_path, data_dir, exploration=1.0, temperature=1.0, search_budget=1600, verbose=False):
    state = State.empty()
    player_name = 'AlphaConnect (%s)' % model_path.split('/')[-1]
    player = AlphaConnectPlayer(model_path, player_name, exploration, temperature, search_budget=search_budget)
    observers = [AlphaConnectSerializer(data_dir)]
    if verbose:
        observers.append(ConsoleObserver())
    game = TwoPlayerGame(state, player, player, observers)
    game.play()
    player.clear_session()
    return game


def is_first_model(model_dir):
    model_files = list(list_files(model_dir, '.h5'))
    return len(model_files) == 0


def latest_model_path(model_dir):
    model_files = list(sorted(list_files(model_dir, '.h5')))
    model_iteration = len(model_files)
    model_path = os.path.abspath(os.path.join(model_dir, model_files[-1]))
    return model_iteration - 1, model_path


def new_model_path(model_dir):
    model_files = list(list_files(model_dir, '.h5'))
    model_iteration = len(model_files)
    model_path = os.path.abspath(os.path.join(model_dir, '%6.6d.h5' % model_iteration))
    return model_iteration, model_path
