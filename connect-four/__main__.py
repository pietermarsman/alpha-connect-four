from game import TwoPlayerGame
from observer import ConsoleObserver
from player import ConsolePlayer, GreedyPlayer
from state import State

state = State.empty()
player1 = GreedyPlayer('White')
player2 = ConsolePlayer('Brown')
observers = [ConsoleObserver()]

game = TwoPlayerGame(state, player1, player2, observers)

game.play()