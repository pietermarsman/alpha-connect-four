from game import TwoPlayerGame
from player import RandomPlayer
from state import ConnectFour3D, Stone


def test_game():
    state = ConnectFour3D()
    player1 = RandomPlayer()
    player2 = RandomPlayer()

    game = TwoPlayerGame(state, player1, player2)

    game._turn()
    game._turn()

    number_of_stones = sum((stone is not Stone.NONE for stone in game.current_state.stones.values()))
    assert 2 == number_of_stones
