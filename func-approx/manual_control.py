from enum import IntEnum

import keyboard

from envs.treasure_hunt import TreasureHunt


class ActionTypes(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3


def get_act():
    key = keyboard.read_key()
    act = ActionTypes[key]
    return act


def main():
    env = TreasureHunt(dict(horizon=100, seed=0))
    env.reset()
    env.render()
    done = False
    tot_reward = 0
    while not done:
        env.render()
        act = get_act()
        _, reward, done, _ = env.step(act)
        print(reward)
        tot_reward += reward

    print(tot_reward)


if __name__ == "__main__":
    main()
