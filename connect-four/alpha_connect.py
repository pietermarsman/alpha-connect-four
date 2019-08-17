from argparse import ArgumentParser

from game import TwoPlayerGame
from observer import ConsoleObserver, AlphaConnectSerializer
from player import AlphaConnectPlayer
from state import State


def simulate(model_path, n, exploration, temperature, budget, verbose):
    state = State.empty()
    player_name = 'AlphaConnect (%s)' % model_path.split('/')[-1]
    player = AlphaConnectPlayer(player_name, model_path, exploration, temperature, budget)
    observers = [ConsoleObserver(verbose, verbose), AlphaConnectSerializer()]

    for _ in range(n):
        # todo use multiple processes
        game = TwoPlayerGame(state, player, player, observers)
        game.play()


def latest_model_path():
    # todo actually get latest model
    return 'models/000000.h5'


if __name__ == '__main__':
    parser = ArgumentParser(description='Simulate games with Alpha Connect MCTS')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')
    parser.add_argument('--model', type=str, help='path to model', default=latest_model_path())
    parser.add_argument('--exploration', type=float, help='exploration constant during tree search', default=1.0)
    parser.add_argument('--temperature', type=float, help='temperature for selecting best action', default=1.0)
    parser.add_argument('--budget', type=int, help='number of milliseconds of tree search per turn', default=5000)
    parser.add_argument('-n', type=int, help='number of games to play', default=1)

    args = parser.parse_args()

    simulate(args.model, args.n, args.exploration, args.temperature, args.budget, args.verbose)
