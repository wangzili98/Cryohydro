from __future__ import annotations

import torch


def station_metrics(
    pred: torch.Tensor,
    obs: torch.Tensor,
    min_count: int = 30,
) -> dict[str, torch.Tensor]:
    pred = pred.squeeze(-1).float()
    obs = obs.squeeze(-1).float()
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    mask_f = mask.to(dtype=pred.dtype)
    pred_safe = torch.where(mask, pred, torch.zeros_like(pred))
    obs_safe = torch.where(mask, obs, torch.zeros_like(obs))

    count = mask_f.sum(dim=0)
    pred_mean = (pred_safe * mask_f).sum(dim=0) / count.clamp_min(1.0)
    obs_mean = (obs_safe * mask_f).sum(dim=0) / count.clamp_min(1.0)
    pred_anom = torch.where(mask, pred - pred_mean.unsqueeze(0), torch.zeros_like(pred))
    obs_anom = torch.where(mask, obs - obs_mean.unsqueeze(0), torch.zeros_like(obs))

    sse = ((pred_safe - obs_safe) ** 2 * mask_f).sum(dim=0)
    obs_ss = (obs_anom.square() * mask_f).sum(dim=0)
    pred_ss = (pred_anom.square() * mask_f).sum(dim=0)
    pred_std = torch.sqrt(pred_ss / count.clamp_min(1.0))
    obs_std = torch.sqrt(obs_ss / count.clamp_min(1.0))

    nse = 1.0 - sse / (obs_ss + 1e-8)
    rmse = torch.sqrt(sse / count.clamp_min(1.0) + 1e-8)
    bias = pred_mean - obs_mean
    bias_pct = bias / (obs_mean.abs() + 1e-8) * 100.0

    corr = (pred_anom * obs_anom * mask_f).sum(dim=0) / (
        torch.sqrt(pred_ss * obs_ss) + 1e-8
    )
    alpha = pred_std / (obs_std + 1e-8)
    beta = pred_mean / (obs_mean + 1e-8)
    kge = 1.0 - torch.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    valid = (count >= min_count) & (obs_ss > 1e-8)
    nan = torch.full_like(nse, float("nan"))
    return {
        "count": count,
        "nse": torch.where(valid, nse, nan),
        "kge": torch.where(valid, kge, nan),
        "corr": torch.where(valid, corr, nan),
        "alpha": torch.where(valid, alpha, nan),
        "beta": torch.where(valid, beta, nan),
        "rmse": torch.where(valid, rmse, nan),
        "pred_mean": torch.where(count > 0, pred_mean, nan),
        "obs_mean": torch.where(count > 0, obs_mean, nan),
        "pred_std": torch.where(count > 0, pred_std, nan),
        "obs_std": torch.where(count > 0, obs_std, nan),
        "bias": torch.where(count > 0, bias, nan),
        "bias_pct": torch.where(count > 0, bias_pct, nan),
    }


def mean_metric(pred: torch.Tensor, obs: torch.Tensor, metric: str, min_count: int = 30) -> torch.Tensor:
    values = station_metrics(pred, obs, min_count=min_count)[metric]
    valid = torch.isfinite(values)
    if not bool(valid.any()):
        return torch.full((), float("nan"), device=values.device, dtype=values.dtype)
    return values[valid].mean()


def nse_star_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    train_std: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    pred = pred.squeeze(-1).float()
    obs = obs.squeeze(-1).float()
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    mask_f = mask.to(dtype=pred.dtype)
    pred_safe = torch.where(mask, pred, torch.zeros_like(pred))
    obs_safe = torch.where(mask, obs, torch.zeros_like(obs))

    count = mask_f.sum(dim=0)
    se_sum = ((pred_safe - obs_safe) ** 2 * mask_f).sum(dim=0)
    mse = se_sum / count.clamp_min(1.0)

    train_std = train_std.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    station_loss = mse / (train_std + eps) ** 2
    valid_station = (count > 0).to(dtype=pred.dtype)
    return (station_loss * valid_station).sum() / valid_station.sum().clamp_min(1.0)


def mean_station_nse(pred: torch.Tensor, obs: torch.Tensor, min_count: int = 30) -> torch.Tensor:
    return mean_metric(pred, obs, "nse", min_count=min_count)


def mean_station_kge(pred: torch.Tensor, obs: torch.Tensor, min_count: int = 30) -> torch.Tensor:
    return mean_metric(pred, obs, "kge", min_count=min_count)
