from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = f"{Path(__file__).resolve().parents[1].name}.train"

from ..config.cryohydro_config import (
    CryoHydroConfig, DATA_DIR, INPUT_VERSION, FORCING_NAMES, FORCING_UNITS,
    PHY_FORCINGS, PHY_ATTRIBUTES, STATIC_ATTRIBUTES, ROUTING_ATTRIBUTES,
    DYNAMIC_PARAMS, STATIC_PARAMS, ROUTING_PARAMS,
)
from .physical_model import CryoHydroModel
from .losses import nse_star_loss, station_metrics


def get_device(name: str) -> torch.device:
    device = torch.device("cuda:0" if name == "auto" and torch.cuda.is_available()
                          else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return device


def build_topology(path: Path, basin_ids: np.ndarray) -> dict:
    positions = {int(b): i for i, b in enumerate(basin_ids)}
    downstream = np.full(len(basin_ids), -1, dtype="int64")
    area = np.full(len(basin_ids), np.nan, dtype="float32")
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bid = int(row["HYBAS_ID"])
            if bid not in positions:
                continue
            if bid in seen:
                raise ValueError(f"Duplicate topology basin: {bid}")
            seen.add(bid)
            i = positions[bid]
            downstream[i] = positions.get(int(row["NEXT_DOWN"] or 0), -1)
            area[i] = float(row["UP_AREA"]) * 1e6
    if len(seen) != len(basin_ids) or not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("Topology must provide positive upstream area for every basin")
    indegree = np.bincount(downstream[downstream >= 0], minlength=len(basin_ids))
    levels, current, visited = [], np.flatnonzero(indegree == 0).tolist(), 0
    while current:
        levels.append(current)
        visited += len(current)
        following = []
        for i in current:
            j = downstream[i]
            if j >= 0:
                indegree[j] -= 1
                if indegree[j] == 0:
                    following.append(int(j))
        current = following
    if visited != len(basin_ids):
        raise ValueError("The river network contains a cycle")
    return dict(downstream_indices=downstream.tolist(), topo_levels=levels,
                route_area=area.tolist())


def _standardize(values, mean, std):
    return (values - np.asarray(mean, dtype="float32")[:, None, None]) / np.asarray(std, dtype="float32")[:, None, None]


def load_data(data_dir: Path, config: CryoHydroConfig, normalization=None) -> dict:
    """Load the ready-to-use release schema; no legacy adapters or unit conversion."""
    with np.load(data_dir / "dynamic_data.npz", allow_pickle=False) as z:
        dynamic = z["data"].astype("float32")
        names = z["variables"].astype(str).tolist()
        dates = z["dates"].astype("U10")
        basin_ids = z["basin_ids"].astype("int64")
        metadata = json.loads(str(z["config"]))
    if names != FORCING_NAMES or metadata.get("units") != FORCING_UNITS:
        raise ValueError("Forcing must contain the 11 configured variables in their documented units")
    days = dates.astype("datetime64[D]")
    if len(days) < config.sequence_length or np.isnat(days).any() or not np.all(np.diff(days) == np.timedelta64(1, "D")):
        raise ValueError("Forcing needs a contiguous sequence of at least 730 days")
    if len(set(basin_ids.tolist())) != len(basin_ids) or dynamic.shape != (len(names), len(basin_ids), len(dates)):
        raise ValueError("Invalid forcing basin IDs or dimensions")
    if not np.isfinite(dynamic).all():
        raise ValueError("Forcing cannot contain missing values")
    with np.load(data_dir / "static_data.npz", allow_pickle=False) as z:
        static = z["data"].astype("float32")
        static_names = z["variables"].astype(str).tolist()
        if not np.array_equal(z["basin_ids"], basin_ids):
            raise ValueError("Static and forcing basin IDs/order differ")
    if static.shape != (len(static_names), len(basin_ids)) or len(set(static_names)) != len(static_names):
        raise ValueError("Invalid static data dimensions/variable names")
    required = set(PHY_ATTRIBUTES + STATIC_ATTRIBUTES + ROUTING_ATTRIBUTES)
    if not required.issubset(static_names):
        raise ValueError(f"Missing static attributes: {sorted(required - set(static_names))}")
    selected = static[[static_names.index(n) for n in required]]
    if not np.isfinite(selected).all():
        raise ValueError("Required static attributes must be finite")

    with np.load(data_dir / "target_data.npz", allow_pickle=False) as z:
        target = z["data"].astype("float32")
        target_names = z["variables"].astype(str).tolist()
        target_dates = z["dates"].astype("U10")
        station_ids = z["basin_ids"].astype("int64")
        station_codes = z["station_codes"].astype(str)
        target_meta = json.loads(str(z["config"]))
    if target_names != ["runoff", "streamflow"] or target_meta.get("units") != {"runoff": "mm/day", "streamflow": "m3/s"}:
        raise ValueError("Target variables/units differ from the release schema")
    if target.shape != (2, len(station_ids), len(target_dates)) or len(station_codes) != len(station_ids):
        raise ValueError("Invalid station target dimensions")
    if len(set(target_dates.tolist())) != len(target_dates) or len(set(station_codes.tolist())) != len(station_codes):
        raise ValueError("Target dates and station codes must be unique")
    if np.isinf(target).any() or (target[np.isfinite(target)] < 0).any():
        raise ValueError("Invalid observations; use NaN for missing values")
    positions = {str(day): i for i, day in enumerate(dates)}
    if not set(target_dates.tolist()).issubset(positions):
        raise ValueError("Target dates fall outside forcing dates")
    target_indices = np.asarray([positions[str(day)] for day in target_dates])
    y = np.full((len(dates), len(station_ids), 1), np.nan, dtype="float32")
    y[target_indices, :, 0] = target[target_names.index("streamflow")].T
    basin_position = {int(b): i for i, b in enumerate(basin_ids)}
    if not set(station_ids.tolist()).issubset(basin_position):
        raise ValueError("Station outlets are missing from the basin network")
    station_indices = [basin_position[int(b)] for b in station_ids]

    if normalization is None:
        # This limited example is one training sequence, without held-out research periods.
        mean = dynamic.mean(axis=(1, 2), dtype="float64")
        std = dynamic.std(axis=(1, 2), dtype="float64")
        static_mean = static.mean(axis=1, dtype="float64")
        static_std = static.std(axis=1, dtype="float64")
        normalization = dict(variables=names, mean=mean.tolist(), std=np.where(std > 0, std, 1).tolist(),
                             static_variables=static_names, static_mean=static_mean.tolist(),
                             static_std=np.where(static_std > 0, static_std, 1).tolist(),
                             input_version=INPUT_VERSION, period=dates[[0, -1]].tolist())
    if normalization.get("input_version") != INPUT_VERSION or normalization.get("variables") != names or normalization.get("static_variables") != static_names:
        raise ValueError("Checkpoint normalization does not match the input schema")
    for key, size in [("mean", len(names)), ("std", len(names)), ("static_mean", len(static_names)), ("static_std", len(static_names))]:
        vector = np.asarray(normalization[key])
        if vector.shape != (size,) or not np.isfinite(vector).all() or ("std" in key and (vector <= 0).any()):
            raise ValueError(f"Invalid normalization: {key}")
    normalized = _standardize(dynamic, normalization["mean"], normalization["std"]).transpose(2, 1, 0)
    static_normalized = (static - np.asarray(normalization["static_mean"], dtype="float32")[:, None]) / np.asarray(normalization["static_std"], dtype="float32")[:, None]
    def attrs(values, chosen):
        return values[[static_names.index(n) for n in chosen]].T.copy()
    obs = y[config.warm_up:, :, 0]
    mask = np.isfinite(obs)
    count = mask.sum(axis=0)
    mean = np.where(mask, obs, 0).sum(axis=0) / np.maximum(count, 1)
    variance = np.square(np.where(mask, obs - mean, 0)).sum(axis=0) / np.maximum(count, 1)
    train_std = np.where((count > 0) & (variance > 0), np.sqrt(variance), 1).astype("float32")
    topology = build_topology(data_dir / "basinatlas_topology.csv", basin_ids)
    return dict(x_phy=dynamic[[names.index(n) for n in PHY_FORCINGS]].transpose(2, 1, 0).copy(),
                x_dynamic=normalized.copy(), c_phy=attrs(static, PHY_ATTRIBUTES),
                c_norm=attrs(static_normalized, STATIC_ATTRIBUTES), r_norm=attrs(static_normalized, ROUTING_ATTRIBUTES),
                y=y, dates=dates, basin_ids=basin_ids, station_ids=station_ids, station_codes=station_codes,
                station_indices=np.asarray(station_indices), train_std=train_std,
                normalization=normalization, topology=topology, target_dates=target_dates)


def tensors_to_device(arrays: dict, device: torch.device) -> dict:
    return {name: torch.as_tensor(arrays[name], device=device, dtype=torch.long if name == "station_indices" else torch.float32)
            for name in ["x_phy", "x_dynamic", "c_phy", "c_norm", "r_norm", "y", "station_indices", "train_std"]}


def make_model(config: CryoHydroConfig, topology: dict, device: torch.device) -> CryoHydroModel:
    phy = dict(**topology, forcings=PHY_FORCINGS, attributes=PHY_ATTRIBUTES,
               dynamic_params=DYNAMIC_PARAMS, static_params=STATIC_PARAMS, routing_params=ROUTING_PARAMS,
               routing_method="muskingum", routing_substeps=config.routing_substeps,
               routing_x_max=config.routing_x_max, routing_k_cap_hours=config.routing_k_cap_hours,
               output_unit="m3/s", warm_up=config.warm_up, warm_up_states=True,
               nmul=config.nmul, use_glacier=True, use_permafrost=True,
               output_names=["streamflow"], route_components=False)
    nn_config = dict(input_size=len(FORCING_NAMES) + len(STATIC_ATTRIBUTES),
                     static_input_size=len(STATIC_ATTRIBUTES), routing_input_size=len(ROUTING_ATTRIBUTES),
                     hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout,
                     static_hidden_size=config.static_hidden_size, static_dropout=config.static_dropout,
                     routing_hidden_size=config.routing_hidden_size, routing_dropout=config.routing_dropout,
                     nmul=config.nmul, dynamic_params=DYNAMIC_PARAMS, static_params=STATIC_PARAMS,
                     routing_params=ROUTING_PARAMS)
    return CryoHydroModel(dict(phy_model=phy, nn_model=nn_config), device=device).to(device)


def batch(tensors: dict, start: int, end: int) -> dict:
    dynamic = tensors["x_dynamic"][start:end]
    static = tensors["c_norm"].unsqueeze(0).expand(end - start, -1, -1)
    return dict(x_phy=tensors["x_phy"][start:end], c_phy=tensors["c_phy"],
                xc_nn_norm=torch.cat([dynamic, static], dim=-1), c_norm=tensors["c_norm"], r_norm=tensors["r_norm"])


def generate_dynamic(model, inputs, config, track_grad):
    state = None
    pieces = []
    for start in range(0, len(inputs), config.parameter_chunk_days):
        x = inputs[start:start + config.parameter_chunk_days]
        if state is None:
            size = (config.num_layers, x.shape[1], config.hidden_size)
            state = (x.new_zeros(size), x.new_zeros(size))
        def step(value, hidden, cell):
            output, updated = model.generate_dynamic_parameters(value, (hidden, cell), return_state=True)
            return output, updated[0], updated[1]
        if track_grad and config.checkpoint_gradients:
            output, hidden, cell = checkpoint(step, x, state[0], state[1], use_reentrant=False)
        else:
            output, hidden, cell = step(x, state[0], state[1])
        state = (hidden, cell)
        pieces.append(output)
    return torch.cat(pieces)


def training_prediction(model, data, config):
    raw = dict(dynamic=generate_dynamic(model, data["xc_nn_norm"], config, True))
    raw["static"], raw["routing"] = model.generate_static_parameters(data["c_norm"], data["r_norm"])
    phy = model.phy_model
    params = phy._descale_parameters(raw)
    x, attrs = data["x_phy"], data["c_phy"]
    states = phy._initial_states(x, attrs)
    with torch.no_grad():
        states, _ = phy._run(x[:config.warm_up], attrs, states,
                             {name: value[:config.warm_up] for name, value in params.items()})
    state_names, param_names = list(states), list(params)
    outputs = []
    for start in range(config.warm_up, len(x), config.physics_chunk_days):
        end = min(start + config.physics_chunk_days, len(x))
        def step(forcing, attributes, *values):
            state = dict(zip(state_names, values[:len(state_names)]))
            p = dict(zip(param_names, values[len(state_names):]))
            updated, out = phy._run(forcing, attributes, state, p)
            return tuple(updated[name] for name in state_names) + (out["streamflow"],)
        args = (x[start:end], attrs, *[states[n] for n in state_names], *[params[n][start:end] for n in param_names])
        result = checkpoint(step, *args, use_reentrant=False) if config.checkpoint_gradients else step(*args)
        states = dict(zip(state_names, result[:-1]))
        outputs.append(result[-1])
    return torch.cat(outputs)


def train_model(data_dir: Path, config: CryoHydroConfig, device: torch.device) -> Path:
    if config.sequence_length != 730 or config.warm_up != 365:
        raise ValueError("This release retains the existing 730-day sequence and 365-day spin-up")
    if min(config.epochs, config.steps_per_epoch, config.parameter_chunk_days, config.physics_chunk_days, config.nmul) < 1:
        raise ValueError("Epochs, steps, members and chunk lengths must be positive")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    arrays = load_data(data_dir, config)
    observed = np.isfinite(arrays["y"][:, :, 0]).any(axis=1)
    cumulative = np.r_[0, np.cumsum(observed)]
    starts = [i for i in range(len(arrays["dates"]) - config.sequence_length + 1)
              if cumulative[i + config.sequence_length] > cumulative[i + config.warm_up]]
    if not starts:
        raise ValueError("No observed target after the 365-day spin-up; check target dates")
    tensors = tensors_to_device(arrays, device)
    model = make_model(config, arrays["topology"], device)
    optimizer = torch.optim.AdamW([
        dict(params=model.nn_model.parameters(), lr=config.learning_rate),
        dict(params=model.static_model.parameters(), lr=config.learning_rate),
        dict(params=model.routing_model.parameters(), lr=config.learning_rate * config.routing_lr_factor)],
        weight_decay=config.weight_decay)
    output_dir = data_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "CryoHydro.pt"
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for _ in range(config.steps_per_epoch):
            start = random.choice(starts); end = start + config.sequence_length
            inputs = batch(tensors, start, end)
            optimizer.zero_grad(set_to_none=True)
            prediction = training_prediction(model, inputs, config).index_select(1, tensors["station_indices"])
            target = tensors["y"][start + config.warm_up:end]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite streamflow prediction")
            loss = nse_star_loss(prediction, target, tensors["train_std"], config.nse_epsilon)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite training gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
        value = float(np.mean(losses))
        history.append(dict(epoch=epoch, training_nse_star_loss=value))
        print(f"Epoch {epoch}/{config.epochs}: NSE* loss={value:.6g}", flush=True)
        torch.save(dict(model_name="CryoHydro", input_version=INPUT_VERSION,
                        config=asdict(config), normalization=arrays["normalization"],
                        model_state_dict=model.state_dict(), history=history,
                        forcing_names=FORCING_NAMES, phy_attributes=PHY_ATTRIBUTES,
                        static_attributes=STATIC_ATTRIBUTES, routing_attributes=ROUTING_ATTRIBUTES,
                        dynamic_params=DYNAMIC_PARAMS, static_params=STATIC_PARAMS,
                        routing_params=ROUTING_PARAMS, routing_method="muskingum",
                        training_target_period=arrays["target_dates"][[0, -1]].tolist(),
                        note="Limited training example; no independent validation/test period."), path)
    with (data_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(f"Saved checkpoint: {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train the CryoHydro model on package-local data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    config = CryoHydroConfig(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    device = get_device(args.device)
    print(f"CryoHydro | {device} | sequence=730, spin-up=365 | Muskingum | NSE*", flush=True)
    train_model(args.data_dir.resolve(), config, device)


if __name__ == "__main__":
    main()
