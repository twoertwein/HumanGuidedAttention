
import torch
from python_tools.ml.neural import Attenuated_Modality_Experts


@torch.jit.script
def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.to(device=x.device, dtype=x.dtype)
    not_nan = ~torch.isnan(x)
    n_not_nan = not_nan.sum(dim=0)
    rs = []

    n_nans = torch.unique(n_not_nan).view(-1, 1)
    for index, n_nan in zip(n_not_nan == n_nans.view(-1, 1), n_nans):
        n_nan = n_nan.item()
        if n_nan < 2:
            continue
        mask = not_nan & index
        x_ = x[mask].view(n_nan, -1)
        y_ = y[mask].view(n_nan, -1)
        index = (x_.std(dim=0) > 1e-8) & (y_.std(dim=0) > 1e-8)
        if not index.any():
            rs.append(torch.zeros(1, device=x.device, dtype=x.dtype))
            continue
        x_ = x_[:, index]
        y_ = y_[:, index]
        covariance = torch.cov(torch.cat([x_, y_], dim=1).t())
        size = x_.shape[1]
        arange = torch.arange(size)
        diagonal = covariance.diagonal().sqrt()
        rs.append(
            covariance[arange, arange + size] / (diagonal[:size] * diagonal[size:])
        )
    return torch.cat(rs)


class Attenuated_Modality_Experts_Human(Attenuated_Modality_Experts):
    def __init__(self, *args, guided: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided = guided

    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss_value = super().loss(scores, ground_truth, meta)
        importance = loss_value  # avoid unbound variable
        if self.guided:
            importance = torch.cat(
                [meta["meta_Visual"], meta["meta_Acoustic"], meta["meta_Language"]],
                dim=1,
            )

        if not self.guided or not (importance == importance).any():
            return loss_value
        normed_attenuation = meta["meta_all_normed_attenuations"]

        correlations = 1 - pearson_correlation(importance.detach(), normed_attenuation)
        return loss_value + correlations.mean() * self.attenuation_lambda
