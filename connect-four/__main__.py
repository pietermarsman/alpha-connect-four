from argparse import ArgumentParser

from alpha_connect import simulate, optimize
from classifier import train_new_model
from game import TwoPlayerGame
from observer import ConsoleObserver
from player import ConsolePlayer, GreedyPlayer
from state import State


def play_game(args):
    state = State.empty()
    player1 = GreedyPlayer('White')
    player2 = ConsolePlayer('Brown')
    observers = [ConsoleObserver()]

    game = TwoPlayerGame(state, player1, player2, observers)

    game.play()


def train_model(args):
    train_new_model(args.data_path, args.model_path)


def simulate_alpha_connect_game(args):
    simulate(args.model_path, args.n, args.data_path)


def optimize_alpha_connect(args):
    # todo this command has increased memory usage over time
    optimize(args.rounds, args.games)


parser = ArgumentParser(description='A game of connect four in three dimensions')
subparsers = parser.add_subparsers()

parser_play = subparsers.add_parser('play', help='play an interactive game of connect four in three dimensions')
parser_play.set_defaults(func=play_game)

parser_train = subparsers.add_parser('train', help='train a neural network on played games')
parser_train.add_argument('data_path', help='path to data from previous games')
parser_train.add_argument('model_path', help='path where to store the serialized neural network')
parser_train.set_defaults(func=train_model)

parser_simulate = subparsers.add_parser('simulate', help='simulate a new game played by a neural network')
parser_train.add_argument('data_path', help='path to data from previous games')
parser_simulate.add_argument('model_path', type=str, help='path to a serialized neural network')
parser_simulate.add_argument('-n', type=int, help='number of games to play', default=1)
parser_simulate.set_defaults(func=simulate_alpha_connect_game)

parser_optimize = subparsers.add_parser('optimize', help='iteratively optimize neural network by self-play')
parser_optimize.add_argument('--rounds', type=int, help='number of games to play', default=1)
parser_optimize.add_argument('--games', type=int, help='number of games to play', default=2500)
parser_optimize.set_defaults(func=optimize_alpha_connect)

args = parser.parse_args()
args.func(args)
