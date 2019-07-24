from game import TwoPlayerGame
from observer import ConsoleObserver
from player import RandomPlayer, ConsolePlayer
from state import ConnectFour3D


state = ConnectFour3D()
player1 = RandomPlayer('White')
player2 = ConsolePlayer('Brown')
observers = [ConsoleObserver()]

game = TwoPlayerGame(state, player1, player2, observers)

game.play()