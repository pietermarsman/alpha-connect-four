import os
import time
from itertools import cycle
from multiprocessing.pool import Pool

from classifier import train_new_model, write_model
from game import TwoPlayerGame
from observer import GameStatePrinter, AlphaConnectSerializer, AlphaConnectPrinter
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
        write_model(model, model_path)
        time.sleep(wait)


def simulate_continuously(model_dir, data_dir, processes, search_budget):
    with Pool(processes) as p:
        for _ in p.imap_unordered(simulate_once_with_newest_model, cycle([(model_dir, data_dir, search_budget)])):
            pass


def simulate_once_with_newest_model(args):
    model_dir, data_dir, search_budget = args
    model_iteration, model_path = latest_model_path(model_dir)
    model_data_dir = os.path.join(data_dir, '%6.6d' % model_iteration)
    simulate_once(model_path, model_data_dir, search_budget=search_budget)


def simulate_once(model_path, data_dir=None, exploration=1.0, temperature=1.0, search_budget=1600, verbose=False):
    state = State.empty()
    player_name = 'AlphaConnect (%s)' % model_path.split('/')[-1]
    player = AlphaConnectPlayer(model_path, player_name, exploration, temperature, search_budget=search_budget,
                                self_play=True)
    observers = []
    if data_dir is not None:
        observers.append(AlphaConnectSerializer(data_dir))
    if verbose:
        observers.append(AlphaConnectPrinter())
        observers.append(GameStatePrinter())
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
