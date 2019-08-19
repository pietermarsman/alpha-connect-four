import os

from classifier import train_new_model
from game import TwoPlayerGame
from observer import ConsoleObserver, AlphaConnectSerializer
from player import AlphaConnectPlayer
from state import State

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(_ROOT_DIR, 'models')
DATA_DIR = os.path.join(_ROOT_DIR, 'data')


def optimize(rounds, games_per_round):
    if is_first_model(MODEL_DIR):
        model_iteration, model_path = new_model_path(MODEL_DIR)
        train_new_model(None, model_path)
    else:
        model_iteration, model_path = latest_model_path(MODEL_DIR)

    for _ in range(rounds):
        data_dir = os.path.join(DATA_DIR, '%6.6d' % model_iteration)
        os.makedirs(data_dir, exist_ok=True)
        simulate(model_path, games_per_round, data_dir)

        model_iteration, model_path = new_model_path(MODEL_DIR)
        train_new_model(data_dir, model_path)


def is_first_model(model_dir):
    model_files = list_files(model_dir, '.h5')
    return len(model_files) == 0


def latest_model_path(model_dir):
    model_files = list_files(model_dir, '.h5')
    model_iteration = len(model_files)
    model_path = os.path.abspath(os.path.join(model_dir, model_files[-1]))
    return model_iteration - 1, model_path


def simulate(model_path, n, data_dir, exploration=1.0, temperature=1.0, budget=2000, verbose=False):
    while count_games(data_dir) < n:
        # todo use multiple processes
        state = State.empty()
        player_name = 'AlphaConnect (%s)' % model_path.split('/')[-1]
        player = AlphaConnectPlayer(player_name, model_path, exploration, temperature, budget)
        observers = [ConsoleObserver(verbose, verbose), AlphaConnectSerializer(data_dir)]
        game = TwoPlayerGame(state, player, player, observers)
        game.play()
        player.clear_session()


def count_games(data_dir):
    data_files = list_files(data_dir, '.json')
    return len(data_files)


def new_model_path(model_dir):
    model_files = list_files(model_dir, '.h5')
    model_iteration = len(model_files)
    model_path = os.path.abspath(os.path.join(model_dir, '%6.6d.h5' % model_iteration))
    return model_iteration, model_path


def list_files(path, ext):
    files = os.listdir(path)
    model_files = list(sorted([f for f in files if f.endswith(ext)]))
    return model_files
