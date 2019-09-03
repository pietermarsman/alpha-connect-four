from game import TwoPlayerGame
from observer import GameStatePrinter
from player import RandomPlayer, GreedyPlayer, MiniMaxPlayer
from state import State, Color


def test_game():
    state = State.empty()
    player1 = RandomPlayer('player 1')
    player2 = RandomPlayer('player 2')

    game = TwoPlayerGame(state, player1, player2)

    game._turn()
    game._turn()

    number_of_stones = sum((stone is not Color.NONE for stone in game.current_state.stones.values()))
    assert 2 == number_of_stones


def test_greedy_is_better_than_random():
    state = State.empty()
    player1 = RandomPlayer('player 1')
    player2 = GreedyPlayer('player 2')

    game = TwoPlayerGame(state, player1, player2)
    game.play()

    assert Color.BROWN == game.current_state.winner


def test_minimax_is_better_than_greedy():
    state = State.empty()
    player1 = GreedyPlayer('player 1')
    player2 = MiniMaxPlayer('player 2')

    game = TwoPlayerGame(state, player1, player2)
    game.play()

    assert Color.BROWN == game.current_state.winner


def test_no_errors_when_playing_and_printing_games():
    for i in range(100):
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        observers = [GameStatePrinter(show_action_history=True)]
        game = TwoPlayerGame(State.empty(), player1, player2, observers)
        game.play()
