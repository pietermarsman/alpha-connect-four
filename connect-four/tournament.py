import json
import os
import re
from itertools import cycle
from multiprocessing.pool import Pool
from random import choice

import numpy as np
from pystan import StanModel

from game import TwoPlayerGame
from observer import GameWinnerSerializer
from player import RandomPlayer, GreedyPlayer, MiniMaxPlayer, MonteCarloPlayer, AlphaConnectPlayer, Player
from state import State
from util import list_files


def tournament_continuously(tournament_dir, model_dir, processes, first_player_name_filter, first_player_kwargs_filter,
                            second_player_name_filter, second_player_kwargs_filter):
    os.makedirs(tournament_dir, exist_ok=True)

    with Pool(processes, maxtasksperchild=10) as p:
        for _ in p.imap_unordered(play_random_opponenents_game_once, cycle([(
                tournament_dir, model_dir, first_player_name_filter, first_player_kwargs_filter,
                second_player_name_filter, second_player_kwargs_filter)])):
            pass


def play_random_opponenents_game_once(args):
    tournament_dir, model_dir, first_player_name_filter, first_player_kwargs_filter, second_player_name_filter, \
    second_player_kwargs_filter = args
    player1 = random_player(model_dir, first_player_name_filter, first_player_kwargs_filter)
    player2 = random_player(model_dir, second_player_name_filter, second_player_kwargs_filter)
    print(repr(player1), 'vs', repr(player2))
    state = State.empty()
    observers = [GameWinnerSerializer(tournament_dir)]
    game = TwoPlayerGame(state, player1, player2, observers)
    game.play()


def random_player(model_dir, name_filter=None, kwargs_filter=None) -> Player:
    players = list_players(model_dir)
    if name_filter is not None:
        players = filter(lambda player_kwargs: re.match(name_filter, player_kwargs[0].__name__), players)
    if kwargs_filter is not None:
        players = filter(lambda player_kwargs: re.match(kwargs_filter, json.dumps(player_kwargs[1])), players)

    player_cls, player_kwargs = choice(list(players))
    return player_cls(**player_kwargs)


def list_players(model_dir):
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
    model_files = list(sorted(list_files(model_dir, '.h5')))
    for model_file in model_files:
        if model_file.endswith('0.h5'):
            players += [
                (AlphaConnectPlayer, {'model_path': model_file, 'search_budget': 1600}),
            ]
    return players


def bayes_tournament_elo(tournament_dir: str):
    games = read_games(tournament_dir)

    elo_code = """
    data {
        int<lower=0> N; // number of games
        int<lower=0> M; // number of players
        int white[N];
        int brown[N];
        int win[N];
    }
    parameters {
        real advantage;
        vector[M] elo;
    }
    model {
        advantage ~ normal(0, 1);
        elo ~ normal(0, 1);
        win ~ bernoulli_logit(elo[white] - elo[brown] + advantage);
    }
    """

    players = list(set([game['white'] for game in games] + [game['brown'] for game in games]))
    white_index = [players.index(game['white']) for game in games]
    brown_index = [players.index(game['brown']) for game in games]
    winner_index = [int(game['winner'] == game['white']) for game in games]

    data = {
        'N': len(games),
        'M': len(players),
        'white': np.array(white_index) + 1,
        'brown': np.array(brown_index) + 1,
        'win': np.array(winner_index)
    }
    elo_model = StanModel(model_code=elo_code)
    fit = elo_model.sampling(data)

    values = fit.extract()
    advantage = np.quantile(values['advantage'], [.025, .5, .975])
    player_games = {player: sum(game['white'] == player or game['brown'] == player for game in games) for player in
                    players}
    player_wins = {player: sum(game['winner'] == player for game in games) for player in players}
    rating = {player: np.quantile(values['elo'], 0.5, axis=0)[player_i] for player_i, player in enumerate(players)}
    lower = {player: np.quantile(values['elo'], 0.025, axis=0)[player_i] for player_i, player in enumerate(players)}
    upper = {player: np.quantile(values['elo'], 0.975, axis=0)[player_i] for player_i, player in enumerate(players)}
    best = {players[player_i]: wins / values['elo'].shape[0] for player_i, wins in
            zip(*np.unique(values['elo'].argmax(axis=1), return_counts=True))}

    score_board = sorted(rating.items(), key=lambda player_rating: -player_rating[1])
    print('Advantage of starting player: %.2f (>%.2f, <%.2f)' % (advantage[1], advantage[0], advantage[2]))
    print('%6s  %6s  %6s  |  %6s  %6s  %6s  |  %6s  |  %s' %
          ('games', 'wins', 'losses', 'lower', 'median', 'upper', 'best', 'player'))
    for player, player_rating in score_board:
        losses = player_games[player] - player_wins[player]
        print('%6s  %6d  %6d  |  %6.2f  %6.2f  %6.2f  |  %5.0f%%  |  %s' %
              (player_games[player], player_wins[player], losses, lower[player], player_rating, upper[player],
               best.get(player, 0.0) * 100, player))


def read_games(tournament_dir):
    game_files = list_files(tournament_dir, '.json')
    games = []
    for game_file in game_files:
        with open(game_file, 'r') as fin:
            games.append(json.load(fin))
    return games
