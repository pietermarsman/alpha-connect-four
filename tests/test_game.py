from game import TwoPlayerGame
from player import RandomPlayer, GreedyPlayer
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

    assert player2.color == game.current_state.winner
