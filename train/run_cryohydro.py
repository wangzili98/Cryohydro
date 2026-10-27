from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = f"{Path(__file__).resolve().parents[1].name}.train"

from ..config.cryohydro_config import DATA_DIR, INPUT_VERSION, CryoHydroConfig
from . import train_cryohydro as core

OUTPUT_NAMES = ["streamflow", "Q0", "Q1", "Q2", "AET", "SWE", "root_water",
                "sub_water", "groundwater", "freeze_depth", "thaw_depth", "glac_storage", "glac_frac"]
OUTPUT_UNITS = dict(streamflow="m3/s", Q0="mm/day", Q1="mm/day", Q2="mm/day", AET="mm/day",
                    SWE="mm", root_water="mm", sub_water="mm", groundwater="mm",
                    freeze_depth="mm", thaw_depth="mm", glac_storage="m3 water equivalent", glac_frac="1")


@torch.no_grad()
def simulate(model, tensors, config, chunk_days, save_parameters=True):
    """Carry both LSTM state and every physical/routing state across chunks."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be positive")
    model.eval()
    phy = model.phy_model
    phy.output_names = OUTPUT_NAMES.copy()
    phy.route_components = False
    raw_static, raw_routing = model.generate_static_parameters(tensors["c_norm"], tensors["r_norm"])
    length, basins = tensors["x_phy"].shape[:2]
    outputs = np.empty((len(OUTPUT_NAMES), basins, length), dtype="float32")
    dynamic = np.empty((len(core.DYNAMIC_PARAMS), basins, length), dtype="float32") if save_parameters else None
    nn_state, physical_state = None, None
    static_values = routing_values = None
    for start in range(0, length, chunk_days):
        end = min(start + chunk_days, length)
        data = core.batch(tensors, start, end)
        raw_dynamic, nn_state = model.generate_dynamic_parameters(data["xc_nn_norm"], nn_state, return_state=True)
        parameters = phy._descale_parameters(dict(dynamic=raw_dynamic, static=raw_static, routing=raw_routing))
        if physical_state is None:
            physical_state = phy._initial_states(data["x_phy"], data["c_phy"])
            static_values = np.stack([parameters[n][0].mean(dim=-1).cpu().numpy() for n in core.STATIC_PARAMS])
            routing_values = np.stack([parameters[n][0].mean(dim=-1).cpu().numpy() for n in core.ROUTING_PARAMS])
        physical_state, out = phy._run(data["x_phy"], data["c_phy"], physical_state, parameters)
        for i, name in enumerate(OUTPUT_NAMES):
            value = out[name].squeeze(-1)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite output {name} at days {start}:{end}")
            outputs[i, :, start:end] = value.float().cpu().numpy().T
        if dynamic is not None:
            for i, name in enumerate(core.DYNAMIC_PARAMS):
                dynamic[i, :, start:end] = parameters[name].mean(dim=-1).float().cpu().numpy().T
        nn_state = tuple(t.detach() for t in nn_state)
        physical_state = {name: value.detach() for name, value in physical_state.items()}
        print(f"Output days {start + 1}-{end}/{length}", flush=True)
    return dict(outputs=outputs, dynamic_parameters=dynamic,
                static_parameters=static_values, routing_parameters=routing_values)


def run_model(data_dir: Path, checkpoint_path: Path, device, chunk_days=None):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("model_name") != "CryoHydro" or saved.get("input_version") != INPUT_VERSION:
        raise ValueError("Checkpoint is not from this CryoHydro release")
    config = CryoHydroConfig(**saved["config"])
    arrays = core.load_data(data_dir, config, saved["normalization"])
    tensors = core.tensors_to_device(arrays, device)
    model = core.make_model(config, arrays["topology"], device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    result = simulate(model, tensors, config, chunk_days or config.output_chunk_days)
    station_prediction = torch.from_numpy(result["outputs"][0].T[:, arrays["station_indices"]]).unsqueeze(-1)
    obs = torch.from_numpy(arrays["y"])
    metrics = core.station_metrics(station_prediction[config.warm_up:], obs[config.warm_up:])
    metric_names = list(metrics)
    metrics_array = np.stack([metrics[name].cpu().numpy() for name in metric_names])
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(model_name="CryoHydro", input_version=INPUT_VERSION, config=asdict(config),
                    routing_method="muskingum", loss="NSE*", normalization=saved["normalization"],
                    output_units=OUTPUT_UNITS, parameter_networks=dict(dynamic="LSTM", static="KAN", routing="KAN"),
                    forcing_period=arrays["dates"][[0, -1]].tolist(), target_period=arrays["target_dates"][[0, -1]].tolist(),
                    warm_up_days=config.warm_up, score_period="released training observations after spin-up",
                    note="Illustrative trained outputs, not independent validation/test or paper results.",
                    output_shape="(variable, basin, time)", target_shape="(station, time)",
                    dynamic_parameter_shape="(parameter, basin, time); ensemble member mean",
                    static_parameter_shape="(parameter, basin); ensemble member mean",
                    routing_parameter_note="parX is the dimensionless Muskingum weight; parK is the learned normalized K control, not hours.")
    path = output_dir / "CryoHydro_outputs.npz"
    np.savez_compressed(path, **result, variables=np.asarray(OUTPUT_NAMES),
                        dates=arrays["dates"], basin_ids=arrays["basin_ids"],
                        target=arrays["y"][:, :, 0].T, station_codes=arrays["station_codes"],
                        station_basin_ids=arrays["station_ids"],
                        station_prediction=result["outputs"][0, arrays["station_indices"], :],
                        station_metrics=metrics_array, metric_names=np.asarray(metric_names),
                        dynamic_parameter_names=np.asarray(core.DYNAMIC_PARAMS),
                        static_parameter_names=np.asarray(core.STATIC_PARAMS),
                        routing_parameter_names=np.asarray(core.ROUTING_PARAMS),
                        config=np.asarray(json.dumps(metadata)))
    with (output_dir / "station_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station", *metric_names])
        for i, name in enumerate(arrays["station_codes"]):
            writer.writerow([name, *metrics_array[:, i].tolist()])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved outputs: {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate CryoHydro outputs from its saved checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-days", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    data_dir = args.data_dir.resolve()
    checkpoint_path = args.checkpoint or data_dir / "checkpoints" / "CryoHydro.pt"
    run_model(data_dir, checkpoint_path, core.get_device(args.device), args.chunk_days)


if __name__ == "__main__":
    main()
