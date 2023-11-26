import torch
from torch import nn as nn


class TokenSelect(nn.Module):
    def __init__(
        self,
        expansion_step: list = [0, 100, 200],
        keep_rate: list = [0.5, 0.75, 1.0],
        initialization_keep_rate: float = 0.25,
        expansion_multiple_stage: int = 2,
        distance: str = "cosine",
    ):
        super().__init__()
        self.expansion_stage = 0
        self.sparse_inference = False

        self.expansion_step = expansion_step
        self.total_expansion_stage = len(expansion_step)
        self.initialization_keep_rate = initialization_keep_rate

        self.expansion_keep_rate = []
        for i in range(len(keep_rate)):
            if i == 0:
                self.expansion_keep_rate.append(keep_rate[i] - initialization_keep_rate)
            else:
                self.expansion_keep_rate.append(keep_rate[i] - keep_rate[i - 1])

        self.final_keep_rate = keep_rate[-1]
        self.expansion_multiple_stage = expansion_multiple_stage

        self.distance = distance

    def update_current_stage(self, epoch: int):
        import bisect

        expansion_stage = bisect.bisect_right(self.expansion_step, epoch)
        self.expansion_stage = expansion_stage

    def get_score(self, a: torch.Tensor, b: torch.Tensor):
        if self.distance == "cosine":
            dist = a @ b.transpose(-1, -2)
        elif self.distance == "manhattan":
            dist = torch.sum(
                torch.abs(a.unsqueeze(2) - b.unsqueeze(1)),
                dim=-1,
            )
        elif self.distance == "euclidean":
            dist = torch.sqrt(torch.sum((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, dim=-1))
        else:
            raise Exception("Wrong distance!", self.distance)
        return dist

    def token_initialization(self, token: torch.Tensor):
        x = int((self.token_num - 1) * self.initialization_keep_rate)
        step = int(1 // self.initialization_keep_rate)
        with torch.no_grad():
            select_index = []
            unselect_index = []
            for i in range(self.token_num - 1):
                if i % step == 0 and len(select_index) < x:
                    select_index.append(i)
                else:
                    unselect_index.append(i)
            select_index = (
                torch.tensor(select_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                x,
                token.shape[2],
            )
            unselect_index = (
                torch.tensor(unselect_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                token.shape[1] - x,
                token.shape[2],
            )

        select_token = token.gather(dim=1, index=select_index)
        unselect_token = token.gather(dim=1, index=unselect_index)

        assert select_token.shape[1] + unselect_token.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"
        assert select_index.shape[1] + unselect_index.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"

        return (select_token, select_index), (unselect_token, unselect_index)

    def token_expansion(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,
    ):
        for stage in range(1, self.expansion_stage + 1):
            if stage == self.total_expansion_stage:
                expansion_token_num = int(
                    (self.token_num - 1) * self.final_keep_rate
                ) - int(
                    (self.token_num - 1)
                    * (
                        self.initialization_keep_rate
                        + sum([self.expansion_keep_rate[i] for i in range(stage - 1)])
                    )
                )
            else:
                expansion_token_num = int(
                    (self.token_num - 1) * self.expansion_keep_rate[stage - 1]
                )

            for k in range(1, self.expansion_multiple_stage + 1):
                if k == self.expansion_multiple_stage:
                    multiple_expansion_token_num = expansion_token_num - (
                        self.expansion_multiple_stage - 1
                    ) * (expansion_token_num // self.expansion_multiple_stage)
                else:
                    multiple_expansion_token_num = (
                        expansion_token_num // self.expansion_multiple_stage
                    )

                with torch.no_grad():
                    select_token_norm = select_token / select_token.norm(
                        dim=-1, keepdim=True
                    )
                    unselect_token_norm = unselect_token / unselect_token.norm(
                        dim=-1, keepdim=True
                    )

                    scores = self.get_score(unselect_token_norm, select_token_norm)

                    node_max, node_idx = scores.max(dim=-1)
                    edge_idx = node_max.argsort(dim=-1, descending=False)[..., None]

                    add_node_index = edge_idx[..., :multiple_expansion_token_num, :]
                    unadd_node_index = edge_idx[..., multiple_expansion_token_num:, :]

                add_index = unselect_index.gather(
                    dim=1,
                    index=add_node_index.expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )
                add_token = unselect_token.gather(
                    dim=1,
                    index=add_node_index.expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )
                select_index = torch.cat([select_index, add_index], dim=1)
                select_token = torch.cat([select_token, add_token], dim=1)

                unselect_index = unselect_index.gather(
                    dim=1,
                    index=unadd_node_index.expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )
                unselect_token = unselect_token.gather(
                    dim=1,
                    index=unadd_node_index.expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )

                assert select_token.shape[1] + unselect_token.shape[1] == (
                    self.token_num - 1
                ), "Wrong shape!"
                assert select_index.shape[1] + unselect_index.shape[1] == (
                    self.token_num - 1
                ), "Wrong shape!"
        return (select_token, select_index), (unselect_token, unselect_index)

    def token_merge(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,
        mode="mean",
    ):
        rest_token_num = unselect_token.shape[1]

        with torch.no_grad():
            select_token_norm = select_token / select_token.norm(dim=-1, keepdim=True)
            unselect_token_norm = unselect_token / unselect_token.norm(
                dim=-1, keepdim=True
            )
            # scores = unselect_token_norm @ select_token_norm.transpose(-1, -2)
            scores = self.get_score(unselect_token_norm, select_token_norm)

            node_max, node_idx = scores.max(dim=-1)
            merge_unselect_node_index = node_idx[..., None]

        select_token = select_token.scatter_reduce(
            dim=1,
            index=merge_unselect_node_index.expand(
                unselect_token.shape[0],
                rest_token_num,
                unselect_token.shape[2],
            ),
            src=unselect_token,
            reduce=mode,
        )

        return (select_token, select_index)

    def token_select(self, x):
        self.token_num = x.shape[1]
        select_index = None
        if (
            self.expansion_stage > 0
            and not (
                self.expansion_stage == self.total_expansion_stage
                and self.final_keep_rate == 1.0
            )
            and self.sparse_inference
        ):
            # separate CLS token from all token
            token_cls = x[..., :1, :]
            # token initialization
            (select_token, select_index), (
                unselect_token,
                unselect_index,
            ) = self.token_initialization(x[..., 1:, :])
            # token expansion
            (select_token, select_index), (
                unselect_token,
                unselect_index,
            ) = self.token_expansion(
                select_token,
                select_index,
                unselect_token,
                unselect_index,
            )
            # token merge
            if unselect_token.shape[1] > 0:
                (select_token, select_index) = self.token_merge(
                    select_token, select_index, unselect_token, unselect_index, "mean"
                )

            x = torch.cat([token_cls, select_token], dim=1)
            cls_index = torch.zeros([x.shape[0], 1, x.shape[2]]).to(
                device=select_index.device
            )
            select_index = select_index + 1
            select_index = torch.cat([cls_index, select_index], dim=1)
            select_index = select_index.long()
            assert x.shape[1] == select_index.shape[1], "Wrong shape!"

        return x, select_index
