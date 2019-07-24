from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from player import Player
from state import ConnectFour3D, Stone


class Observer(object):
    def notify_new_state(self, state: 'ConnectFour3D'):
        pass

    def notify_new_action(self, player: Player, action: Tuple[int, int]):
        pass


class ConsoleObserver(Observer):
    def notify_new_state(self, state: 'ConnectFour3D'):
        print(state)

    def notify_new_action(self, player: Player, action: Tuple[int, int]):
        print('\n%s plays %s' % (player, action))