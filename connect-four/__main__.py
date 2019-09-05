import time
from argparse import ArgumentParser

from alpha_connect import simulate_once, optimize_continuously, optimize_once, \
    simulate_continuously
from game import TwoPlayerGame
from observer import GameStatePrinter, AlphaConnectPrinter
from player import ConsolePlayer, AlphaConnectPlayer
from state import State, Action
from tournament import tournament_continuously, bayes_tournament_elo


def _play_game(args):
    human_player = ConsolePlayer('You')
    computer_player = AlphaConnectPlayer(args.model_path, 'Computer', time_budget=14500)
    observers = [AlphaConnectPrinter(), GameStatePrinter(show_action_history=True)]

    if args.human_first:
        game = TwoPlayerGame(State.empty(), human_player, computer_player, observers)
    else:
        game = TwoPlayerGame(State.empty(), computer_player, human_player, observers)

    if args.actions is not None:
        for action_hex in args.actions:
            game.play_action(game.next_player(), Action.from_hex(action_hex))

    game.play()


def _optimize_once(args):
    optimize_once(args.data_dir, args.model_path, args.max_games)


def _optimize_continuously(args):
    optimize_continuously(args.model_dir, args.data_dir, args.max_games)


def _simulate_once(args):
    simulate_once(args.model_path, args.data_path, verbose=True)


def _simulate_continously(args):
    simulate_continuously(args.model_dir, args.data_dir, args.processes, args.search_budget)


def _timeit_single_search(args):
    print('Preparing player and state')
    player = AlphaConnectPlayer(args.model_path, start_temperature=None, search_budget=10000)
    state = State.empty()
    print('Running search')
    s0 = time.time()
    player.decide(state)
    duration = time.time() - s0
    print('Done search')
    print(player.root)
    print('Total running time: %.2f' % duration)


def _tournament_continuously(args):
    tournament_continuously(args.tournament_dir, args.model_dir, args.processes, args.first_player_name_filter,
                            args.first_player_kwargs_filter, args.second_player_name_filter,
                            args.second_player_kwargs_filter)


def _tournament_elo(args):
    bayes_tournament_elo(args.tournament_dir)


parser = ArgumentParser(description='A game of connect four in three dimensions')
subparsers = parser.add_subparsers()

# play
parser_play = subparsers.add_parser('play', help='play an interactive game of connect four in three dimensions')
parser_play.add_argument('model_path', help='path where the model should be stored')
parser_play.add_argument('--actions', help='actions to play in initial state (default: none)')
parser_play.add_argument('--human_second', help='let console player go second', dest='human_first', default=True,
                         action='store_false')
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
parser_optimize_continuously.add_argument('--max_games', help='maximum number of games to train on', default=50000,
                                          type=int)
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
parser_simulate_continuously.add_argument('--search_budget', type=int, help='number of mcts searches', default=1600)
parser_simulate_continuously.set_defaults(func=_simulate_continously)

# timeit
parser_timeit = subparsers.add_parser('timeit')
parser_timeit.add_argument('model_path', type=str, help='path to a serialized neural network')
parser_timeit.set_defaults(func=_timeit_single_search)

# tournament-continously
parser_tournament_continuously = subparsers.add_parser('tournament-continously',
                                                       help='generate games between randomly chosen players')
parser_tournament_continuously.add_argument('tournament_dir', help='directory where tournament games are stored')
parser_tournament_continuously.add_argument('model_dir', help='directory where alpha connect models are stored')
parser_tournament_continuously.add_argument('--processes', type=int, help='number of cores to use', default=4)
parser_tournament_continuously.add_argument('--first_player_name_filter', help='regex filter for first player name')
parser_tournament_continuously.add_argument('--first_player_kwargs_filter', help='regex filter for first kwargs')
parser_tournament_continuously.add_argument('--second_player_name_filter', help='regex filter for second player name')
parser_tournament_continuously.add_argument('--second_player_kwargs_filter', help='regex filter for second kwargs')
parser_tournament_continuously.set_defaults(func=_tournament_continuously)

# tournament-elo
parser_tournament_elo = subparsers.add_parser('tournament-elo', help='compute elo score for tournament players')
parser_tournament_elo.add_argument('tournament_dir', help='directory where tournament games are stored')
parser.set_defaults(func=_tournament_elo)

args = parser.parse_args()
args.func(args)
