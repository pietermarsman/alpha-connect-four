from argparse import ArgumentParser

from alpha_connect import simulate_once, optimize_continuously, optimize_once, \
    simulate_continuously
from game import TwoPlayerGame
from observer import ConsoleObserver
from player import ConsolePlayer, AlphaConnectPlayer
from state import State


def _play_game(args):
    player2 = ConsolePlayer('You')
    player1 = AlphaConnectPlayer(args.model_path, 'Computer', time_budget=14500)
    observers = [ConsoleObserver()]

    game = TwoPlayerGame(State.empty(), player1, player2, observers)

    game.play()


def _optimize_once(args):
    optimize_once(args.data_dir, args.model_path, args.max_games)


def _optimize_continuously(args):
    optimize_continuously(args.model_dir, args.data_dir, args.max_games)


def _simulate_once(args):
    simulate_once(args.model_path, args.data_path, verbose=True)


def _simulate_continously(args):
    simulate_continuously(args.model_dir, args.data_dir, args.processes)


parser = ArgumentParser(description='A game of connect four in three dimensions')
subparsers = parser.add_subparsers()

# play
parser_play = subparsers.add_parser('play', help='play an interactive game of connect four in three dimensions')
parser_play.add_argument('model_path', help='path where the model should be stored')
parser_play.set_defaults(func=_play_game)

# optimize-once
parser_optimize_once = subparsers.add_parser('optimize-once', help='train a neural network on played games')
parser_optimize_once.add_argument('data_dir', help='directory where data is stored')
parser_optimize_once.add_argument('model_path', help='path where the model should be stored')
parser_optimize_once.add_argument('--max_games', help='maximum number of games to train on', default=50000)
parser_optimize_once.set_defaults(func=_optimize_once)

# optimize-continuously
parser_optimize_continuously = subparsers.add_parser('optimize-continuously',
                                                     help='train neural network continuously and keep best')
parser_optimize_continuously.add_argument('data_dir', help='directory where data is stored')
parser_optimize_continuously.add_argument('model_dir', help='directory where model is stored')
parser_optimize_continuously.add_argument('--max_games', help='maximum number of games to train on', default=50000)
parser_optimize_continuously.set_defaults(func=_optimize_continuously)

# simulate-once
parser_simulate_once = subparsers.add_parser('simulate-once',
                                             help='generate self-play games using existing neural network')
parser_simulate_once.add_argument('model_path', type=str, help='path to a serialized neural network')
parser_simulate_once.add_argument('data_path', help='where to save the generated game')
parser_simulate_once.set_defaults(func=_simulate_once)

# simulate-continuously
parser_simulate_continuously = subparsers.add_parser('simulate-continuously',
                                                     help='generate self-play games using existing neural network')
parser_simulate_continuously.add_argument('model_dir', help='directory where model is stored')
parser_simulate_continuously.add_argument('data_dir', help='directory where data is stored')
parser_simulate_continuously.add_argument('--processes', type=int, help='number of cores to use', default=4)
parser_simulate_continuously.set_defaults(func=_simulate_continously)

args = parser.parse_args()
args.func(args)
