import os
from itertools import cycle
from multiprocessing.pool import Pool
from random import sample

from game import TwoPlayerGame
from observer import GameWinnerSerializer
from player import RandomPlayer, GreedyPlayer, MiniMaxPlayer, MonteCarloPlayer, AlphaConnectPlayer
from state import State
from util import list_files


def tournament_continuously(tournament_dir, model_dir, processes):
    os.makedirs(tournament_dir, exist_ok=True)

    with Pool(processes, maxtasksperchild=10) as p:
        for _ in p.imap_unordered(play_random_opponenents_game_once, cycle([(tournament_dir, model_dir)])):
            pass


def play_random_opponenents_game_once(args):
    data_dir, model_dir = args
    player1, player2 = random_opponents(model_dir)
    state = State.empty()
    observers = [GameWinnerSerializer(data_dir)]
    game = TwoPlayerGame(state, player1, player2, observers)
    game.play()


def random_opponents(model_dir=None):
    players = [
        (RandomPlayer, {}),
        (GreedyPlayer, {}),
        (MiniMaxPlayer, {'depth': 1}),
        (MiniMaxPlayer, {'depth': 2}),
        (MiniMaxPlayer, {'depth': 3}),
        (MonteCarloPlayer, {'budget': 400}),
        (MonteCarloPlayer, {'budget': 800}),
        (MonteCarloPlayer, {'budget': 1600}),
        (MonteCarloPlayer, {'budget': 3200}),
        (MonteCarloPlayer, {'budget': 6400}),
    ]
    if model_dir is not None:
        for model_file in list_files(model_dir, '.h5'):
            players += [
                (AlphaConnectPlayer, {'model_path': model_file, 'search_budget': 1600}),
            ]

    (player1_cls, player1_kwargs), (player2_cls, player2_kwargs) = tuple(sample(players, 2))
    return player1_cls(**player1_kwargs), player2_cls(**player2_kwargs)
