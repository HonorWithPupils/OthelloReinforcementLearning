import torch
import Othello

from matplotlib import pyplot as plt

if __name__ == "__main__":

    n = 8

    env = Othello.Envs(n, 1, if_render=True)
    obs, reward, terminated, _, info = env.reset()

    actor = Othello.Actor(n, "ResNet")
    actor.load_state_dict(
        torch.load(r"model\n=8_beta=0.5_bone=ResNet_win\epoch_bestVsRandom.pth")
    )

    agents = [actor.sample, env.manualAction]  # AI black Random white
    # agents = [env.randomAction, actor.sample] # AI white Munual black
    player = 0

    finish = False

    while not finish:
        env.render()

        agent = agents[player]

        p = "Black" if player == 0 else "White"
        print(f"player: {p}")

        obs, reward, terminated, _, info = env.step(agent(obs), restart=False)

        if player == 0:
            player = 1
        else:
            player = 0

        plt.pause(0.5)

        finish = terminated[0].bool()

    if info["board"].sum() > 0:
        print(f"result: Black win")
    elif info["board"].sum() < 0:
        print(f"result: White win")
    elif info["board"].sum() == 0:
        print(f"result: Draw")

    print("game over")
    input("press any key to exit")
