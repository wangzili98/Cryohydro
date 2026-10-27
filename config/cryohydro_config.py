from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
FORCING_NAMES = ['prcp', 'tmean', 'tmax', 'tmin', 'srad', 'lrad', 'pres', 'shum', 'wind', 'pet', 'doy']
FORCING_UNITS = dict(prcp="mm/day", tmean="degree C", tmax="degree C", tmin="degree C", srad="W/m2", lrad="W/m2", pres="Pa", shum="kg/kg", wind="m/s", pet="mm/day", doy="1")
PHY_FORCINGS = ['prcp', 'tmean', 'tmax', 'pet', 'doy']
PHY_ATTRIBUTES = ['glac_frac', 'glac_storage', 'glac_elev', 'elev_mean', 'sub_area', 'up_area', 'root_depth', 'sub_depth', 'slope', 'ORDER_']
STATIC_ATTRIBUTES = ['sub_area', 'up_area', 'dist_main', 'dist_sink', 'glac_frac', 'glac_storage', 'glac_elev', 'perm_frac', 'active_layer_thickness', 'mean_annual_ground_temp', 'elev_mean', 'elev_min', 'elev_max', 'elev_std', 'slope', 'aspect_sin', 'aspect_cos', 'relief_mean', 'root_depth', 'sub_depth', 'root_ksat', 'root_theta_sat', 'sub_theta_sat', 'root_theta_fc', 'sub_theta_fc', 'root_theta_wilt', 'root_theta_dry', 'root_sand_frac', 'sub_sand_frac', 'clay_frac', 'root_bulk_density', 'sub_bulk_density', 'soc', 'cfvo', 'lithology_carbonate_frac', 'lithology_unconsolidated_frac', 'lithology_hardrock_frac', 'logK', 'porosity', 'forest_frac', 'grass_frac', 'snow_ice_frac']
ROUTING_ATTRIBUTES = ['sub_area', 'up_area', 'dist_main', 'dist_sink', 'slope', 'relief_mean', 'elev_mean']
DYNAMIC_PARAMS = ['parTCritSnow', 'parDDFS', 'parBETA', 'parBETAET', 'parK0', 'parUZL', 'parDDFG']
STATIC_PARAMS = ['parSSC', 'parCFR', 'parN', 'parBFS', 'parBFUZ', 'parBFLZ', 'parFC', 'parLP', 'parC', 'parPERC', 'parK1', 'parK2']
ROUTING_PARAMS = ["parX", "parK"]
INPUT_VERSION = "cryohydro-release-celsius-pa-v1"


@dataclass
class CryoHydroConfig:
    seed: int = 2026
    sequence_length: int = 730
    warm_up: int = 365
    # One optimization epoch is the release demonstration, not paper calibration.
    epochs: int = 1
    steps_per_epoch: int = 1
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.15
    static_hidden_size: int = 128
    static_dropout: float = 0.10
    routing_hidden_size: int = 32
    routing_dropout: float = 0.10
    nmul: int = 8
    learning_rate: float = 2e-4
    routing_lr_factor: float = 0.3
    weight_decay: float = 1e-5
    gradient_clip: float = 0.5
    nse_epsilon: float = 0.1
    parameter_chunk_days: int = 30
    physics_chunk_days: int = 30
    output_chunk_days: int = 90
    checkpoint_gradients: bool = True
    routing_substeps: int = 4
    routing_x_max: float = 0.30
    routing_k_cap_hours: float = 12.0
