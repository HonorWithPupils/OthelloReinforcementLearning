import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, Dict, NoReturn

from Attention import ViTBlock
from CNN import ResNet

plt.ion()


class Envs(gym.Env):
    def __init__(
        self,
        n: int,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        if_render=False,
    ) -> NoReturn:

        self.n = n
        self.batch_size = batch_size

        self.action_space = gym.spaces.Discrete(n * n + 1)
        self.observation_space = gym.spaces.Discrete(n * n)
        self.reward_range = (-1, 1)

        # for algorithm and agent reuse, the board defaults to be relative
        # chess pieces of the next player: 1; the previous player: -1; empty space: 0
        self.__initBoard = torch.zeros((n, n))
        self.__initBoard[n // 2 - 1, n // 2 - 1] = 1
        self.__initBoard[n // 2, n // 2] = 1
        self.__initBoard[n // 2 - 1, n // 2] = -1
        self.__initBoard[n // 2, n // 2 - 1] = -1
        self.__board = torch.zeros((batch_size, n, n))
        self.__board[:] = self.__initBoard

        self.__pass = torch.zeros((batch_size,))
        self.__nextPlayer = torch.ones((batch_size,))  # 1: black; -1: white
        self.__rounds = torch.ones((batch_size,))
        self.__end1 = torch.zeros((batch_size,))
        self.__end2 = torch.zeros((batch_size,))  # each player needs a end

        self.__initTemplate()

        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)

        self.if_render = if_render
        if if_render:
            self.fig = plt.figure(figsize=(self.n, self.n))

    def __initTemplate(self):

        def subTemplate(n):

            state, move, res = [], [], []

            for i in range(n):
                for j in range(i + 1, n - 1):
                    state.append(
                        [100] * i + [1] + [-1] * (j - i) + [0] + [100] * (n - 2 - j)
                    )  # 100 for any
                    move.append([0] * i + [0] + [0] * (j - i) + [1] + [0] * (n - 2 - j))
                    res.append([0] * i + [1] + [1] * (j - i) + [0] + [0] * (n - 2 - j))

            f = lambda x: torch.tensor(x + [i[::-1] for i in x])
            state, move, res = f(state), f(move), f(res)

            return state, move, res

        def rowTemplate(n):

            def rowExpand(x, i, k):
                nt, size = x.shape
                res = torch.ones(nt, size, size) * k
                res[:, i, :] = x
                return res

            state, move, res = subTemplate(n)
            state_row, move_row, res_row = [], [], []

            for i in range(n):
                state_row.append(rowExpand(state, i, 100))
                move_row.append(rowExpand(move, i, 0))
                res_row.append(rowExpand(res, i, 0))

            return (
                torch.cat(state_row, 0),
                torch.cat(move_row, 0),
                torch.cat(res_row, 0),
            )

        def diagTemplate(n):

            def diagExpand(x, i, size, k):
                res = torch.ones(x.shape[0], size, size) * k
                a = res.diagonal(i, dim1=-2, dim2=-1)
                a[:, :] = x
                return res

            state_diag, move_diag, res_diag = [], [], []

            for i in range(3 - n, n - 2):
                l = n - abs(i)
                state, move, res = subTemplate(l)
                state_diag.append(diagExpand(state, i, n, 100))
                move_diag.append(diagExpand(move, i, n, 0))
                res_diag.append(diagExpand(res, i, n, 0))

            return (
                torch.cat(state_diag, 0),
                torch.cat(move_diag, 0),
                torch.cat(res_diag, 0),
            )

        state_r, move_r, res_r = rowTemplate(self.n)
        state_d, move_d, res_d = diagTemplate(self.n)

        self.__templateState = torch.cat(
            [state_r, state_r.mT, state_d, state_d.flip(-1)], 0
        )
        self.__templateMove = torch.cat([move_r, move_r.mT, move_d, move_d.flip(-1)], 0)
        self.__templateRes = torch.cat([res_r, res_r.mT, res_d, res_d.flip(-1)], 0)

    @property
    def board(self) -> torch.Tensor:
        """real board

        Returns:
            torch.Tensor (batch, n, n): real board, 1) black; -1) white
        """
        return (self.__board * self.__nextPlayer[:, None, None]).detach()

    def validPosition(self) -> torch.Tensor:
        """Valid positions for the next move

        Returns:
            torch.Tensor (batch, n*n+1): valid positions for the next player
        """
        with torch.no_grad():
            A = (
                (
                    (self.__board[:, None, :, :] == self.__templateState)
                    | (self.__templateState == 100)
                )
                .all(-1)
                .all(-1)[:, :, None, None]
            )
            A = (A * self.__templateMove).any(1).view(-1, self.n * self.n)
            A = torch.cat([A, (A == 0).all(1)[:, None]], dim=1)

        return A.detach()

    def randomAction(self, *args, **kwargs) -> torch.Tensor:
        """Random action

        Returns:
            action (torch.Tensor: (batch, n*n+1)): the action to take
        """
        valid = self.validPosition()
        idx = torch.multinomial(valid.float(), 1).view(-1)
        return F.one_hot(idx, valid.shape[-1]).detach()

    def manualAction(self, *args, **kwargs) -> torch.Tensor:
        """Manual action:
        The player manually enters the number of the next move in a playable position on the 1st board of the batch.
        The move on the rest of boards are random.

        Returns:
            action (torch.Tensor: (batch, n*n+1)): the action to take
        """

        valid = self.validPosition()
        idx = torch.multinomial(valid.float(), 1).view(-1)
        move = F.one_hot(idx, valid.shape[-1]).detach()

        valid0 = valid[0]
        idx0 = torch.cumsum(valid0, dim=-1) * valid[0]

        options = [str(idx0[i].item()) for i in range(len(idx0)) if valid0[i]]

        if valid0[-1] == 1:
            print("There is no playing position, only an empty move. ")
        else:
            user_input = ""

            input_message = "Pick an option move:\n"

            for item in options:
                input_message += f"{item}\t"

            while user_input not in options:
                if user_input != "":
                    user_input = input(
                        input_message
                        + "\nYour input was invalid. Plese, pick an option move: "
                    )

                else:
                    user_input = input(input_message + "\nYour choice: ")

            move0 = idx0 == int(user_input)

            move[0] = move0

        return move

    @torch.no_grad()
    def step(self, action: torch.Tensor, restart: bool = True) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        None,
        Dict[str, torch.Tensor],
    ]:
        """Process one move

        Args:
            position (torch.Tensor: (batch, n*n+1)): Position of this move and whether to pass (batch, n*n+1)
            restart (bool, optional): Whether to restart the games after finishing. Defaults to True.
        Returns:
            next observation & valid (Tuple[torch.Tensor: (batch, n, n), torch.Tensor: (batch, n, n)]):
                next observation: the relative board after this move (the next player: 1)
                valid: the valid position for the next move
            reward (torch.Tensor: (batch,)): the result of game (+1, -1).
            terminated (torch.Tensor: (batch,)): Whether the env is terminated.
            truncated (None): truncation-free
            info ({'board': torch.Tensor (batch, n, n), # real board
                   'nextPlayer': torch.Tensor (batch,), # 1: black; -1: white
                   'rounds': torch.Tensor (batch,), # the number of rounds
                   'score': torch.Tensor (batch,), # score
            })
        """

        with torch.no_grad():
            position = action[:, :-1].view(-1, self.n, self.n)
            ifpass = action[:, -1]

            A = (
                (
                    (self.__board[:, None, :, :] == self.__templateState)
                    | (self.__templateState == 100)
                )
                .all(-1)
                .all(-1)
            )
            B = (position[:, None, :, :] == self.__templateMove).all(-1).all(-1)

            valid = A & B  # (batch, nt)

            tmp = (valid[:, :, None, None] * self.__templateRes).sum(1)  # (batch, s, s)

            self.__board = (tmp == 0) * self.__board + (tmp != 0) * tmp + position

            # Two consecutive passes mean the game is ended
            end = self.__pass * ifpass
            self.__pass = ifpass

            # rounds += 1
            self.__rounds = self.__rounds + 1 * (self.__end2 == 0)

            # reward != 0 only when the game just end
            score = self.__board.sum(-1).sum(-1)
            reward = (
                ((score < 0).float() - (score > 0).float()) * end * (self.__end2 == 0)
            )

            # each player needs a end
            self.__end2 = self.__end1 * end

            self.__end1 = end

            # change the next player and the relative board
            self.__nextPlayer = -self.__nextPlayer
            self.__board = -self.__board

            if restart:
                # reset when the game is ended
                self.__nextPlayer = (
                    self.__end2 == False
                ) * self.__nextPlayer + self.__end2 * 1
                self.__pass = (self.__end2 == False) * self.__pass + self.__end2 * 0
                self.__board = (self.__end2 == False)[
                    :, None, None
                ] * self.__board + self.__end2[:, None, None] * self.__initBoard
                self.__rounds = (self.__end2 == False) * self.__rounds + self.__end2 * 1
                self.__end1 = (self.__end2 == False) * self.__end1 + self.__end2 * 0
                self.__end2 = (self.__end2 == False) * self.__end2 + self.__end2 * 0

            info = {
                "board": self.board,
                "nextPlayer": self.__nextPlayer.detach(),
                "rounds": self.__rounds.detach(),
                "score": (score * self.__nextPlayer).detach(),
            }

        return (
            (self.__board.detach(), self.validPosition()),
            reward.detach(),
            end.detach(),
            None,
            info,
        )

    @torch.no_grad()
    def reset(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        None,
        Dict[str, torch.Tensor],
    ]:
        """Reset the game

        Returns:
            next observation & valid (Tuple[torch.Tensor: (batch, n, n), torch.Tensor: (batch, n, n)]):
                next observation: the relative board after this move (the next player: 1)
                valid: the valid position for the next move
            reward (torch.Tensor: (batch,)): the result of game (+1, -1).
            terminated (torch.Tensor: (batch,)): Whether the env is terminated.
            truncated (None): truncation-free
            info ({'board': torch.Tensor (batch, n, n), # real board
                   'nextPlayer': torch.Tensor (batch,), # 1: black; -1: white
                   'rounds': torch.Tensor (batch,), # the number of rounds
                   'score': torch.Tensor (batch,), # score
            })
        """

        self.__nextPlayer[:] = 1
        self.__pass[:] = 0
        self.__rounds[:] = 1
        self.__board[:] = self.__initBoard

        info = {
            "board": self.board,
            "nextPlayer": self.__nextPlayer.detach(),
            "rounds": self.__rounds.detach(),
            "score": (self.__board.sum(-1).sum(-1) * self.__nextPlayer).detach(),
        }

        return (
            (self.__board.detach(), self.validPosition()),
            self.__pass.detach(),
            self.__pass.detach(),
            None,
            info,
        )

    def render(self, i: int = 0) -> NoReturn:
        if not self.if_render:
            return

        valid = self.validPosition()[i]

        idx = torch.cumsum(valid, axis=-1) * valid
        idx = idx[:-1].view(self.n, self.n).cpu().int().numpy()
        valid = valid[:-1].view(self.n, self.n).cpu().int().numpy()
        board = self.board[i].cpu().int().numpy()

        black = board == 1
        white = board == -1

        fig = self.fig

        fig.clf()

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#d7dec7")

        for i in range(self.n):
            for j in range(self.n):
                if black[i, j]:
                    ax.scatter(i, j, c="black", marker="o", s=3600)
                elif white[i, j]:
                    ax.scatter(i, j, c="white", marker="o", s=3600)
                elif valid[i, j]:
                    ax.scatter(i, j, c="grey", marker=f"${idx[i, j]}$", s=1200)

        for i in range(self.n):
            ax.plot([-0.5, self.n - 0.5], [-0.5 + i, -0.5 + i], c="black")
            ax.plot([-0.5 + i, -0.5 + i], [-0.5, self.n - 0.5], c="black")

        ax.set_xlim([-0.5, self.n - 0.5])
        ax.set_ylim([-0.5, self.n - 0.5])

        ax.set_xticks([])
        ax.set_yticks([])

        fig.show()

    def close(self) -> NoReturn: ...


class Actor(torch.nn.Module):
    def __init__(self, n: int, bone: str, **kwargs) -> NoReturn:

        super().__init__()

        if not kwargs:
            if bone == "ViT":
                kwargs = {
                    "img_size": n,
                    "patch_size": 1,
                    "depth": 10,
                    "dim": 512,
                    "mlp_dim": 1024,
                    "heads": 16,
                    "dim_head": 32,
                    "channels": 1,
                }
            elif bone == "ResNet":
                kwargs = {
                    "dims": [1, 32, 64, 64, 32, 1],
                    "kernel_size": 5,
                }

        self.n = n

        if bone == "ViT":
            self.block = ViTBlock(img_size=n, **kwargs)
        elif bone == "ResNet":
            self.block = ResNet(**kwargs)

        self.linear = torch.nn.Linear(n * n, 1)

    def forward(self, input: Tuple[torch.Tensor]) -> torch.Tensor:

        x, validMask = input
        validMask = validMask.to(torch.bool)

        # input
        # b,1,n,n -> b,1,n,n
        x = self.block(x[:, None])

        # add pass
        # b,1,n,n -> b,n*n+1
        x = x.view(x.size(0), -1)
        ifpass = self.linear(x)
        x = torch.cat((x, ifpass), dim=1)

        # valid mask
        negInf = torch.finfo(x.dtype).min
        x = x * validMask + negInf * ~validMask

        x = F.softmax(x, dim=1)

        return x

    def sample(self, input: Tuple[torch.Tensor]) -> torch.Tensor:
        x = self(input).max(-1)[1].view(-1)
        x = F.one_hot(x, num_classes=self.n * self.n + 1)

        return x


class Critic(torch.nn.Module):
    def __init__(self, n: int, bone: str, **kwargs) -> NoReturn:

        super().__init__()

        if not kwargs:
            if bone == "ViT":
                kwargs = {
                    "img_size": n,
                    "patch_size": 1,
                    "depth": 10,
                    "dim": 512,
                    "mlp_dim": 1024,
                    "heads": 16,
                    "dim_head": 32,
                    "channels": 1,
                }
            elif bone == "ResNet":
                kwargs = {
                    "dims": [1, 32, 64, 64, 32, 1],
                    "kernel_size": 5,
                }

        self.n = n

        if bone == "ViT":
            self.block = ViTBlock(img_size=n, **kwargs)
        elif bone == "ResNet":
            self.block = ResNet(**kwargs)

        self.linear = torch.nn.Linear(n * n, 1)

    def forward(self, input: Tuple[torch.Tensor]) -> torch.Tensor:

        x, validMask = input

        # input
        # b,1,n,n -> b,1,n,n
        x = self.block(x[:, None])

        # calculate predicted reward
        # b,1,n,n -> b,n*n -> b,
        x = x.view(x.size(0), -1)
        x = self.linear(x).view(x.size(0))

        return x
