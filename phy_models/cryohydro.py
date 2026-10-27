from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


class CryoHydro(nn.Module):
    """Differentiable cryosphere model.

    Soil moisture and groundwater response follow the HBV SM-SUZ-SLZ structure.
    Snow, glacier melt and Stefan freeze/thaw are implemented directly in this
    class. Tensor convention is sequence first: [time, basin, feature].
    """

    def __init__(self, config: dict[str, Any] | None = None, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.config = config or {}
        self.device = torch.device(device or self.config.get("device", "cpu"))
        self.eps = float(self.config.get("eps", 1e-6))
        self.warm_up = int(self.config.get("warm_up", 365))
        self.warm_up_states = bool(self.config.get("warm_up_states", True))
        self.debug_nan = bool(self.config.get("debug_nan", False))
        self.output_unit = str(self.config.get("output_unit", "mm")).lower()
        self.route_components = bool(self.config.get("route_components", False))
        self.variables = list(self.config.get("forcings", ["prcp", "tmean", "tmax", "pet", "doy"]))
        self.attribute_names = list(self.config.get("attributes", []))
        self.glacier_volume_exp = 1.375
        self.glacier_lapse_rate = float(self.config.get("glacier_lapse_rate", 0.0065))
        self.glacier_update_doy = float(self.config.get("glacier_update_doy", 273.0))
        self.glacier_frac_min = float(self.config.get("glacier_frac_min", 0.01))
        self.glacier_storage_ratio_max = float(self.config.get("glacier_storage_ratio_max", 2.0))
        self.routing_method = str(self.config.get("routing_method", "sphy")).lower()
        if self.routing_method not in {"muskingum", "sphy"}:
            raise ValueError("routing_method must be 'muskingum' or 'sphy'")
        self.sphy_k_max = float(self.config.get("sphy_k_max", 0.95))
        if not 0.0 < self.sphy_k_max < 1.0:
            raise ValueError("sphy_k_max must be strictly between 0 and 1")
        self.routing_x_max = float(self.config.get("routing_x_max", 0.30))
        self.routing_substeps = int(self.config.get("routing_substeps", 4))
        if self.routing_substeps < 1:
            raise ValueError("routing_substeps must be positive")
        self.routing_k_cap_hours = float(self.config.get("routing_k_cap_hours", 12.0))
        if self.routing_method == "muskingum" and self.routing_k_cap_hours < 24.0 / self.routing_substeps / (2.0 * (1.0 - self.routing_x_max)):
            raise ValueError("routing_k_cap_hours is below the stable K lower bound")
        self.nmul = int(self.config.get("nmul", 1))
        if self.nmul > 1:
            self.register_buffer(
                "freeze_heterogeneity_q",
                torch.linspace(-1.0, 1.0, self.nmul, device=self.device, dtype=torch.float32).view(1, -1),
                persistent=False,
            )
        else:
            self.register_buffer(
                "freeze_heterogeneity_q",
                torch.zeros(1, 1, device=self.device, dtype=torch.float32),
                persistent=False,
            )
        self.use_glacier = bool(self.config.get("use_glacier", True))
        self.use_permafrost = bool(self.config.get("use_permafrost", True))
        self.use_freeze_heterogeneity = bool(self.config.get("use_freeze_heterogeneity", False))
        self.freeze_heterogeneity_strength = float(self.config.get("freeze_heterogeneity_strength", 0.35))
        self.min_season_days = int(self.config.get("min_season_days", 60))
        self.thermal_column_depth_mm = float(
            self.config.get("thermal_column_depth_mm", 4000.0)
        )
        if self.thermal_column_depth_mm <= 0.0:
            raise ValueError("thermal_column_depth_mm must be positive")
        self.default_dynamic_params = [
            "parTCritSnow", "parDDFS", "parDDFG",
            "parBETA", "parBETAET", "parK0", "parUZL",
        ]
        self.default_static_params = [
            "parSSC",
            "parCFR", "parN", "parBFS", "parBFUZ", "parBFLZ",
            "parFC", "parLP", "parC", "parPERC", "parK1", "parK2",
        ]
        self.default_routing_params = ["parK"] if self.routing_method == "sphy" else ["parX", "parK"]
        self.dynamic_params = list(self.config.get("dynamic_params", self.default_dynamic_params))
        self.static_params = list(self.config.get("static_params", self.default_static_params))
        self.routing_params = list(self.config.get("routing_params", self.default_routing_params))
        if self.routing_method == "sphy" and self.routing_params != ["parK"]:
            raise ValueError("SPHY routing requires routing_params=['parK']; parK is daily retention, not Muskingum K")
        self.dy_drop = float(self.config.get("dy_drop", 0.0))
        self.default_output_names = [
            "streamflow", "Q0", "Q1", "Q2", "routed_surfaceflow",
            "routed_interflow", "routed_baseflow", "AET", "SWE",
            "snowpack_land", "snowpack_glac", "meltwater_land", "meltwater_glac",
            "root_water", "sub_water", "groundwater",
            "freeze_depth", "thaw_depth", "frozen_thickness", "cold_index", "warm_index",
            "season_freeze_max", "residual_frozen_thickness",
            "glac_storage", "glac_frac", "acc_glac_melt",
            "prcp", "tmean", "tmax", "temp_glac", "pet", "rain_land", "rain_glac",
            "snow_land", "snow_glac", "land_rain_recharge", "land_snow_recharge",
            "glacier_rain_recharge", "glacier_snow_recharge", "glacier_melt_recharge",
            "land_snow_out",
            "land_snow_melt", "glacier_snow_melt", "land_refreeze", "glacier_refreeze",
            "soil_water_input", "land_recharge", "uz_recharge",
            "rain_eff", "root_perc", "gw_recharge",
            "sm_freeze_frac", "uz_freeze_frac", "lz_freeze_frac",
            "sm_mobile_frac", "uz_mobile_frac", "lz_mobile_frac", "root_excess", "uz_above_threshold", "caprise",
            "snow_to_glacier", "glac_mass_balance", "soil_wetness", "evap_factor",
        ]
        self.output_names = list(self.config.get("output_names", self.default_output_names))
        downstream_tensor = torch.as_tensor(self.config.get("downstream_indices"), dtype=torch.long)
        route_area_tensor = torch.as_tensor(self.config.get("route_area"), dtype=torch.float32)
        self.route_level_names = []
        all_level_indices = []
        for i, level in enumerate(self.config.get("topo_levels", [])):
            idx = torch.as_tensor(level, dtype=torch.long)
            down = downstream_tensor.index_select(0, idx)
            valid = down >= 0
            idx_name = f"route_level_{i}_idx"
            valid_name = f"route_level_{i}_valid"
            down_name = f"route_level_{i}_down_valid"
            valid_idx_name = f"route_level_{i}_valid_idx"
            self.register_buffer(idx_name, idx, persistent=False)
            self.register_buffer(valid_name, valid, persistent=False)
            self.register_buffer(down_name, down[valid], persistent=False)
            self.register_buffer(valid_idx_name, torch.nonzero(valid, as_tuple=False).flatten(), persistent=False)
            self.route_level_names.append((idx_name, valid_name, down_name, valid_idx_name))
            all_level_indices.append(idx)
        if all_level_indices:
            self.register_buffer("route_all_indices", torch.cat(all_level_indices), persistent=False)
        else:
            self.register_buffer("route_all_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("route_area_tensor", route_area_tensor, persistent=False)
        self.parameter_bounds = OrderedDict(
            [
                ("parTCritSnow", (-2.5, 1.0)),
                ("parTCritGlac", (-1.0, 2.0)),
                ("parTCritSoil", (-5.0, 3.0)),
                ("parDDFS", (1.0, 6.0)),
                ("parSSC", (0.03, 0.20)),
                ("parCFR", (0.0, 0.10)),
                ("parDDFG", (1.0, 8.0)),
                ("parN", (0.40, 1.0)),
                ("parBFS", (0.5, 4.0)),
                ("parBFUZ", (0.5, 4.0)),
                ("parBFLZ", (0.5, 4.0)),
                ("parFC", (50.0, 1000.0)),
                ("parBETA", (1.0, 6.0)),
                ("parLP", (0.2, 1.0)),
                ("parBETAET", (0.3, 5.0)),
                ("parC", (0.0, 1.0)),
                ("parPERC", (0.0, 10.0)),
                ("parUZL", (0.0, 100.0)),
                ("parK0", (0.05, 0.9)),
                ("parK1", (0.01, 0.5)),
                ("parK2", (0.001, 0.2)),
                ("parQ0Mult", (0.1, 0.999)),
                ("parQ1Mult", (0.1, 0.999)),
                ("parQ2Mult", (0.1, 0.999)),
                ("parX", (0.0, 0.30)),
                ("parK", (0.0, self.sphy_k_max if self.routing_method == "sphy" else 1.0)),
            ]
        )
        self.param_names = list(self.parameter_bounds.keys())
        self.learned_param_names = self.dynamic_params + self.static_params + self.routing_params
        unknown = sorted(set(self.learned_param_names) - set(self.param_names))
        if unknown:
            raise ValueError(f"Unknown CryoHydro parameters: {unknown}")
        self.fixed_parameters = self._default_fixed_parameters()
        self.fixed_parameters.update(self.config.get("fixed_parameters", {}))
        self.learnable_param_count = (len(self.dynamic_params) + len(self.static_params)) * self.nmul + len(self.routing_params)

    def forward(self, data_dict: dict[str, torch.Tensor], parameters: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        forcing = data_dict["x_phy"].to(self.device)
        attrs = data_dict.get("c_phy")
        attrs = attrs.to(self.device) if attrs is not None else None
        params = self._descale_parameters(parameters)
        states = self._initial_states(forcing, attrs)

        warm_up = self.warm_up if self.warm_up_states else 0
        if warm_up > 0:
            warm_params = {name: value[:warm_up] for name, value in params.items()}
            with torch.no_grad():
                states, _ = self._run(forcing[:warm_up], attrs, states, warm_params)

        run_params = {name: value[warm_up:] for name, value in params.items()}
        states, out = self._run(forcing[warm_up:], attrs, states, run_params)
        if not self.warm_up_states and self.warm_up > 0:
            out = {key: value[self.warm_up:] for key, value in out.items()}
        return out

    def _descale_parameter(self, raw_value: torch.Tensor, name: str) -> torch.Tensor:
        u = torch.sigmoid(raw_value)
        lo, hi = self.parameter_bounds[name]
        default = min(max(float(self.fixed_parameters[name]), lo), hi)
        below_default = lo + (default - lo) * (u / 0.5)
        above_default = default + (hi - default) * ((u - 0.5) / 0.5)
        return torch.where(u < 0.5, below_default, above_default)

    def _descale_parameters(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_dynamic = raw["dynamic"].to(self.device)
        raw_static = raw["static"].to(self.device)
        raw_routing = raw["routing"].to(self.device)
        out: dict[str, torch.Tensor] = {}
        n_steps, n_batch = raw_dynamic.shape[:2]
        raw_dynamic = raw_dynamic.view(n_steps, n_batch, len(self.dynamic_params), self.nmul)
        raw_static = raw_static.view(n_batch, len(self.static_params), self.nmul)

        for name in self.param_names:
            if name not in self.dynamic_params:
                if name in self.static_params:
                    i = self.static_params.index(name)
                    value = self._descale_parameter(raw_static[:, i], name).unsqueeze(0).expand(n_steps, -1, -1)
                elif name in self.routing_params:
                    i = self.routing_params.index(name)
                    value = self._descale_parameter(raw_routing[:, i], name).unsqueeze(0).expand(n_steps, -1)
                else:
                    value = torch.full(
                        (n_steps, n_batch, self.nmul),
                        float(self.fixed_parameters[name]),
                        dtype=raw_dynamic.dtype,
                        device=raw_dynamic.device,
                    )
                out[name] = value
                continue

            i = self.dynamic_params.index(name)
            value = self._descale_parameter(raw_dynamic[:, :, i], name)
            sta = value[-1:].repeat(value.shape[0], 1, 1)
            if self.training and self.dy_drop > 0:
                mask = torch.bernoulli(torch.full_like(value, self.dy_drop))
                value = value * (1.0 - mask) + sta * mask
            out[name] = value
        return out

    def _default_fixed_parameters(self) -> dict[str, float]:
        return {
            "parTCritSnow": -1.0,
            "parTCritGlac": 0.0,
            "parTCritSoil": -0.5,
            "parDDFS": 2.5,
            "parSSC": 0.12,
            "parCFR": 0.05,
            "parDDFG": 4.0,
            "parN": 0.65,
            "parBFS": 2.0,
            "parBFUZ": 2.0,
            "parBFLZ": 2.0,
            "parFC": 300.0,
            "parBETA": 2.0,
            "parLP": 0.7,
            "parBETAET": 1.0,
            "parC": 0.1,
            "parPERC": 1.0,
            "parUZL": 30.0,
            "parK0": 0.1,
            "parK1": 0.05,
            "parK2": 0.01,
            "parQ0Mult": 0.95,
            "parQ1Mult": 0.90 / 0.95,
            "parQ2Mult": 0.85 / 0.90,
            "parX": 0.20,
            "parK": 0.50,
        }

    def _initial_states(self, forcing: torch.Tensor, attrs: torch.Tensor | None) -> dict[str, torch.Tensor]:
        n_batch = forcing.shape[1]
        attrs_d = self._attributes(attrs, n_batch, forcing.dtype)
        glac_storage = attrs_d["glac_storage"].unsqueeze(-1).expand(-1, self.nmul).clone()
        if not self.use_glacier:
            glac_storage = torch.zeros_like(glac_storage)
        nearzero = torch.full((n_batch, self.nmul), 0.001, device=self.device, dtype=forcing.dtype)
        root_init = torch.full(
            (n_batch, self.nmul),
            float(self.config.get("initial_root_water", 0.3 * self.fixed_parameters["parFC"])),
            device=self.device,
            dtype=forcing.dtype,
        )
        sub_init = torch.full(
            (n_batch, self.nmul),
            float(self.config.get("initial_sub_water", 0.5 * self.fixed_parameters["parUZL"])),
            device=self.device,
            dtype=forcing.dtype,
        )
        gw_init = torch.full(
            (n_batch, self.nmul),
            float(self.config.get("initial_groundwater", 20.0)),
            device=self.device,
            dtype=forcing.dtype,
        )
        q_init = torch.full(
            (n_batch,),
            float(self.config.get("initial_streamflow", 0.0)),
            device=self.device,
            dtype=forcing.dtype,
        )
        return {
            "snow": nearzero.clone(),
            "snow_liq": nearzero.clone(),
            "snow_land": nearzero.clone(),
            "snow_liq_land": nearzero.clone(),
            "snow_glac": torch.zeros_like(nearzero),
            "snow_liq_glac": torch.zeros_like(nearzero),
            "root": root_init,
            "sub": sub_init,
            "gw": gw_init,
            "gw_recharge": torch.zeros(n_batch, self.nmul, device=self.device, dtype=forcing.dtype),
            "baseflow": torch.zeros(n_batch, self.nmul, device=self.device, dtype=forcing.dtype),
            "q": q_init,
            "q_in": q_init.clone(),
            "q_components": q_init.unsqueeze(0).expand(3, -1).clone() / 3.0,
            "q_components_in": q_init.unsqueeze(0).expand(3, -1).clone() / 3.0,
            "route_in": torch.zeros(n_batch, device=self.device, dtype=forcing.dtype),
            "glac_storage": glac_storage,
            "glac_frac": attrs_d["glac_frac"].unsqueeze(-1).expand(-1, self.nmul).clone() if self.use_glacier else torch.zeros_like(glac_storage),
            "acc_glac_melt": torch.zeros(n_batch, self.nmul, device=self.device, dtype=forcing.dtype),
        }

    def _run(
        self,
        forcing: torch.Tensor,
        attrs: torch.Tensor | None,
        states: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        n_steps, n_batch = forcing.shape[:2]
        attrs_d = self._attributes(attrs, n_batch, forcing.dtype)

        series = {name: [] for name in self.output_names}
        streamflow_only = self.output_names == ["streamflow"]
        local_series = []
        component_series = []

        zero = torch.zeros_like(forcing[:, :, 0])
        one = torch.ones_like(forcing[:, :, 0])

        def get_forcing(name: str, fallback: torch.Tensor) -> torch.Tensor:
            if name in self.variables:
                return forcing[:, :, self.variables.index(name)]
            return fallback

        tmean_all = get_forcing("tmean", zero)
        tmax_all = get_forcing("tmax", tmean_all)
        prcp_all = get_forcing("prcp", zero)
        pet_all = get_forcing("pet", zero)
        doy_all = get_forcing("doy", zero)
        par = {name: value[0] for name, value in params.items()}
        routing_coeffs = self._channel_routing_coefficients(par)

        for t in range(n_steps):
            p = torch.clamp(prcp_all[t], min=0.0).unsqueeze(-1).expand(-1, self.nmul)
            temp = tmean_all[t].unsqueeze(-1).expand(-1, self.nmul)
            tmax = tmax_all[t].unsqueeze(-1).expand(-1, self.nmul)
            pet = torch.clamp(pet_all[t], min=0.0).unsqueeze(-1).expand(-1, self.nmul)
            doy = doy_all[t].unsqueeze(-1).expand(-1, self.nmul)
            for name, value in params.items():
                par[name] = value[t]
            self._current_par = par

            glac_frac_t = states["glac_frac"] if self.use_glacier else torch.zeros_like(p)
            snow_to_glacier = torch.zeros_like(p)
            glac_mass_balance = torch.zeros_like(p)
            snow_frac_land = self._soft_gate(par["parTCritSnow"] - temp, tau=0.5)
            land_frac_t = 1.0 - glac_frac_t
            rain_land = p * (1.0 - snow_frac_land) * land_frac_t
            snow_land = p * snow_frac_land * land_frac_t
            temp_glac = self._glacier_temperature(temp, attrs_d) if self.use_glacier else temp
            snow_frac_glac = self._soft_gate(par["parTCritSnow"] - temp_glac, tau=0.5) if self.use_glacier else torch.zeros_like(p)
            rain_glac_depth = p * (1.0 - snow_frac_glac)
            snow_glac_depth = p * snow_frac_glac
            rain_glac = rain_glac_depth * glac_frac_t
            snow_glac = snow_glac_depth * glac_frac_t
            rain = rain_land + rain_glac
            snow = snow_land + snow_glac
            self._debug_check(
                t,
                "partition",
                {
                    "p": p, "temp": temp, "temp_glac": temp_glac, "rain": rain, "snow": snow,
                    "rain_land": rain_land, "rain_glac": rain_glac,
                    "snow_land": snow_land, "snow_glac": snow_glac, "glac_frac": glac_frac_t,
                },
                states,
            )
            (
                states, snow_recharge, snow_melt, refreeze,
                rain_land_soil, rain_glac_ice, old_snow_glac, glacier_snow_out,
                land_snow_melt, glacier_snow_melt, land_refreeze, glacier_refreeze,
            ) = self._snow(
                states, temp, temp_glac, rain_land, snow_land, rain_glac_depth, snow_glac_depth, par
            )
            self._debug_check(
                t,
                "snow",
                {
                    "snow_recharge": snow_recharge,
                    "snow_melt": snow_melt, "refreeze": refreeze,
                    "land_snow_out": snow_recharge, "glacier_snow_out": glacier_snow_out,
                    "land_snow_melt": land_snow_melt, "glacier_snow_melt": glacier_snow_melt,
                    "land_refreeze": land_refreeze, "glacier_refreeze": glacier_refreeze,
                    "snow_to_glacier": snow_to_glacier, "glac_mass_balance": glac_mass_balance,
                },
                states,
            )
            if self.use_glacier:
                states, glacier_melt_recharge, glacier_rain_recharge = self._glacier(
                    states, temp_glac, rain_glac_ice, old_snow_glac, attrs_d, par, glac_frac_t
                )
                glacier_snow_recharge = glacier_snow_out
                glacier_recharge = (
                    glacier_melt_recharge
                    + glacier_rain_recharge
                    + glacier_snow_recharge
                )
                states, snow_to_glacier, glac_mass_balance = self._update_glacier_mass(states, attrs_d, doy)
            else:
                glacier_recharge = torch.zeros_like(p)
                glacier_melt_recharge = torch.zeros_like(p)
                glacier_rain_recharge = torch.zeros_like(p)
                glacier_snow_recharge = torch.zeros_like(p)
            self._debug_check(
                t,
                "glacier",
                {
                    "glacier_recharge": glacier_recharge,
                    "snow_to_glacier": snow_to_glacier,
                    "glac_mass_balance": glac_mass_balance,
                    "glac_frac": states["glac_frac"],
                },
                states,
            )
            pet_eff = pet
            rain_eff = rain_land_soil
            if self.use_permafrost:
                states, z_thaw, z_freeze = self._freeze_thaw(states, temp, attrs_d, par)
            else:
                soil_depth = (attrs_d["root_depth"] + attrs_d["sub_depth"]).unsqueeze(-1).expand(-1, self.nmul)
                z_thaw = soil_depth
                z_freeze = torch.zeros_like(soil_depth)
            self._debug_check(t, "freeze_thaw", {"z_thaw": z_thaw, "z_freeze": z_freeze}, states)
            (
                states, surfaceflow, interflow, baseflow, root_perc, sub_perc,
                soil_aet, infil,
                sm_mobile_frac, uz_mobile_frac, lz_mobile_frac,
                root_excess, uz_above_threshold, caprise,
                water_input, soil_wetness, evap_factor, Q0, Q1, Q2,
                sm_freeze_frac, uz_freeze_frac, lz_freeze_frac,
            ) = self._soil_groundwater(states, rain_eff, snow_recharge, glacier_recharge, pet_eff, z_thaw, z_freeze, attrs_d,
                                       par)
            land_input = rain_eff + snow_recharge
            land_recharge = root_perc + root_excess
            land_rain_recharge = land_recharge * torch.where(
                land_input > self.eps,
                rain_eff / torch.clamp(land_input, min=self.eps),
                torch.zeros_like(land_input),
            )
            land_snow_recharge = land_recharge - land_rain_recharge
            uz_recharge = land_recharge + glacier_recharge
            self._debug_check(
                t,
                "soil_groundwater",
                {
                    "surfaceflow": surfaceflow, "interflow": interflow, "baseflow": baseflow,
                    "root_perc": root_perc, "sub_perc": sub_perc, "AET": soil_aet,
                },
                states,
            )

            runoff = surfaceflow + interflow + baseflow
            self._debug_check(t, "runoff_sum", {"runoff": runoff}, states)
            runoff_mean = self._member_mean(runoff)
            local_series.append(runoff_mean)
            q = runoff_mean  # Replaced by full-sequence routing after production.

            if streamflow_only:
                self._debug_check(t, "routing", {"streamflow": q}, states)
                series["streamflow"].append(q)
            else:
                runoff_components = torch.stack(
                    [
                        self._member_mean(surfaceflow),
                        self._member_mean(interflow),
                        self._member_mean(baseflow),
                    ],
                    dim=0,
                )
                if self.route_components:
                    component_series.append(runoff_components)
                routed_components = torch.zeros_like(runoff_components)

                routed_surfaceflow = routed_components[0]
                routed_interflow = routed_components[1]
                routed_baseflow = routed_components[2]
                values = {
                    "streamflow": q,
                    "Q0": self._member_mean(Q0), "Q1": self._member_mean(Q1), "Q2": self._member_mean(Q2),
                    "routed_surfaceflow": routed_surfaceflow, "routed_interflow": routed_interflow,
                    "routed_baseflow": routed_baseflow,
                    "AET": self._member_mean(soil_aet), "SWE": self._member_mean(states["snow"] + states["snow_liq"]),
                    "snowpack_land": self._member_mean(states["snow_land"]),
                    "snowpack_glac": self._member_mean(states["snow_glac"] * states["glac_frac"]),
                    "meltwater_land": self._member_mean(states["snow_liq_land"]),
                    "meltwater_glac": self._member_mean(states["snow_liq_glac"] * states["glac_frac"]),
                    "root_water": self._member_mean(states["root"]), "sub_water": self._member_mean(states["sub"]),
                    "groundwater": self._member_mean(states["gw"]), "freeze_depth": self._member_mean(z_freeze),
                    "thaw_depth": self._member_mean(z_thaw), "glac_storage": self._member_mean(states["glac_storage"]),
                    "frozen_thickness": self._member_mean(states.get("frozen_thickness", torch.zeros_like(z_freeze))),
                    "cold_index": self._member_mean(states.get("cold_index", torch.zeros_like(z_freeze))),
                    "warm_index": self._member_mean(states.get("warm_index", torch.zeros_like(z_freeze))),
                    "season_freeze_max": self._member_mean(states.get("season_freeze_max", torch.zeros_like(z_freeze))),
                    "residual_frozen_thickness": self._member_mean(states.get("residual_frozen_thickness", torch.zeros_like(z_freeze))),
                    "glac_frac": self._member_mean(states["glac_frac"]), "acc_glac_melt": self._member_mean(states["acc_glac_melt"]),
                    "prcp": self._member_mean(p), "tmean": self._member_mean(temp), "tmax": self._member_mean(tmax),
                    "temp_glac": self._member_mean(temp_glac),
                    "pet": self._member_mean(pet),
                    "rain_land": self._member_mean(rain_land),
                    "rain_glac": self._member_mean(rain_glac),
                    "snow_land": self._member_mean(snow_land), "snow_glac": self._member_mean(snow_glac),
                    "land_rain_recharge": self._member_mean(land_rain_recharge),
                    "land_snow_recharge": self._member_mean(land_snow_recharge),
                    "glacier_rain_recharge": self._member_mean(glacier_rain_recharge),
                    "glacier_snow_recharge": self._member_mean(glacier_snow_recharge),
                    "glacier_melt_recharge": self._member_mean(glacier_melt_recharge),
                    "land_snow_out": self._member_mean(snow_recharge),
                    "land_snow_melt": self._member_mean(land_snow_melt),
                    "glacier_snow_melt": self._member_mean(glacier_snow_melt),
                    "land_refreeze": self._member_mean(land_refreeze),
                    "glacier_refreeze": self._member_mean(glacier_refreeze),
                    "soil_water_input": self._member_mean(water_input),
                    "land_recharge": self._member_mean(land_recharge),
                    "uz_recharge": self._member_mean(uz_recharge),
                    "rain_eff": self._member_mean(rain_eff),
                    "root_perc": self._member_mean(root_perc),
                    "gw_recharge": self._member_mean(states["gw_recharge"]),
                    "sm_freeze_frac": self._member_mean(sm_freeze_frac),
                    "uz_freeze_frac": self._member_mean(uz_freeze_frac),
                    "lz_freeze_frac": self._member_mean(lz_freeze_frac),
                    "sm_mobile_frac": self._member_mean(sm_mobile_frac),
                    "uz_mobile_frac": self._member_mean(uz_mobile_frac),
                    "lz_mobile_frac": self._member_mean(lz_mobile_frac),
                    "root_excess": self._member_mean(root_excess), "uz_above_threshold": self._member_mean(uz_above_threshold), "caprise": self._member_mean(caprise),
                    "snow_to_glacier": self._member_mean(snow_to_glacier),
                    "glac_mass_balance": self._member_mean(glac_mass_balance),
                    "soil_wetness": self._member_mean(soil_wetness), "evap_factor": self._member_mean(evap_factor),
                }
                for key in self.output_names:
                    series[key].append(values[key])

        out = {key: torch.stack(value, dim=0).unsqueeze(-1) for key, value in series.items()}
        states, routed = self._channel_routing_sequence(
            states, torch.stack(local_series), attrs_d, "q", routing_coeffs
        )
        if "streamflow" in out:
            out["streamflow"] = routed.unsqueeze(-1)
        if component_series:
            states, components = self._channel_routing_sequence(
                states, torch.stack(component_series), attrs_d, "q_components", routing_coeffs
            )
            for i, key in enumerate(("routed_surfaceflow", "routed_interflow", "routed_baseflow")):
                if key in out:
                    out[key] = components[:, i].unsqueeze(-1)
        return states, out

    def _attributes(self, attrs: torch.Tensor | None, n_batch: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        defaults = {
            "glac_frac": 0.05, "glac_storage": 0.0, "glac_elev": 0.0, "elev_mean": 0.0,
            "sub_area": 1.0e6, "up_area": 1.0e6,
            "root_depth": 300.0, "sub_depth": 1700.0,
            "slope": 0.05, "ORDER_": 1.0,
        }
        result = {}
        for name, default in defaults.items():
            if attrs is not None and name in self.attribute_names:
                result[name] = attrs[:, self.attribute_names.index(name)].to(dtype=dtype)
            else:
                result[name] = torch.full((n_batch,), float(default), device=self.device, dtype=dtype)
        result["glac_frac"] = torch.clamp(result["glac_frac"], 0.0, 1.0)
        result["glac_storage"] = torch.clamp(result["glac_storage"], min=0.0)
        result["glac_frac"] = torch.where(
            result["glac_frac"] >= self.glacier_frac_min,
            result["glac_frac"],
            torch.zeros_like(result["glac_frac"]),
        )
        result["glac_frac"] = torch.where(
            result["glac_storage"] > 0.0,
            result["glac_frac"],
            torch.zeros_like(result["glac_frac"]),
        )
        result["glac_storage"] = torch.where(
            result["glac_frac"] > 0.0,
            result["glac_storage"],
            torch.zeros_like(result["glac_storage"]),
        )
        result["glac_elev"] = torch.clamp(result["glac_elev"], min=0.0)
        result["elev_mean"] = torch.clamp(result["elev_mean"], min=0.0)
        result["sub_area"] = torch.clamp(result["sub_area"], min=1.0)
        result["up_area"] = torch.clamp(result["up_area"], min=1.0)
        result["root_depth"] = torch.clamp(result["root_depth"], min=300.0, max=300.0)
        result["sub_depth"] = torch.clamp(result["sub_depth"], min=0.0, max=1700.0)
        result["slope"] = torch.clamp(result["slope"], min=0.001)
        result["ORDER_"] = torch.clamp(result["ORDER_"], min=1.0)
        return result

    def _dynamic_glacier_fraction(self, states, attrs):
        glac_frac0 = attrs["glac_frac"].unsqueeze(-1)
        glac_storage0 = attrs["glac_storage"].unsqueeze(-1)

        active = (
            self._soft_gate(glac_frac0 - self.glacier_frac_min, tau=0.002)
            * self._soft_gate(glac_storage0, tau=1.0e5)
            * self._soft_gate(states["glac_storage"], tau=1.0e5)
        )
        storage_ratio = states["glac_storage"] / torch.clamp(glac_storage0, min=self.eps)
        storage_ratio = torch.clamp(storage_ratio, min=1.0e-6, max=self.glacier_storage_ratio_max)
        glac_frac = glac_frac0 * storage_ratio.pow(1.0 / self.glacier_volume_exp)
        glac_frac = torch.clamp(glac_frac, max=1.0)
        states["glac_frac"] = glac_frac * active
        return states, states["glac_frac"]

    def _update_glacier_mass(self, states, attrs, doy):
        update = (torch.round(doy) == self.glacier_update_doy).to(dtype=states["glac_storage"].dtype)
        glac_frac = states["glac_frac"]
        snow_storage = (states["snow_glac"] + states["snow_liq_glac"]) * glac_frac
        mass_balance = update * (snow_storage - states["acc_glac_melt"])
        states["glac_storage"] = torch.clamp(
            states["glac_storage"] + mass_balance / 1000.0 * attrs["sub_area"].unsqueeze(-1),
            min=0.0,
        )
        states["snow_glac"] = states["snow_glac"] * (1.0 - update)
        states["snow_liq_glac"] = states["snow_liq_glac"] * (1.0 - update)
        states["acc_glac_melt"] = states["acc_glac_melt"] * (1.0 - update)
        states["snow"] = states["snow_land"] + states["snow_glac"] * states["glac_frac"]
        states["snow_liq"] = states["snow_liq_land"] + states["snow_liq_glac"] * states["glac_frac"]
        states, _ = self._dynamic_glacier_fraction(states, attrs)
        return states, torch.clamp(mass_balance, min=0.0), mass_balance

    def _glacier_temperature(self, temp, attrs):
        has_glacier_elev = ((attrs["glac_frac"] > 0.0) & (attrs["glac_elev"] > 0.0)).unsqueeze(-1)
        elev_delta = torch.where(
            has_glacier_elev,
            (attrs["glac_elev"] - attrs["elev_mean"]).unsqueeze(-1),
            torch.zeros_like(temp),
        )
        return temp - self.glacier_lapse_rate * elev_delta

    def _snowpack_step(self, snowpack, meltwater, temp, rain_on_snow, snowfall, par, area_fraction=1.0):
        # Stores and inputs share one area basis; scale local degree-day fluxes
        # to that basis (land: basin area; glacier: glacier area).
        old_snow = snowpack
        snowpack = snowpack + snowfall
        melt = torch.clamp(par["parDDFS"] * (temp - par["parTCritSnow"]), min=0.0) * area_fraction
        melt = torch.minimum(melt, snowpack)
        snowpack = snowpack - melt
        meltwater = meltwater + melt + rain_on_snow
        cold_degree = torch.clamp(par["parTCritSnow"] - temp, min=0.0)
        refreeze_pot = par["parCFR"] * par["parDDFS"] * cold_degree * area_fraction
        refreeze = torch.minimum(refreeze_pot, meltwater)
        snowpack = snowpack + refreeze
        meltwater = meltwater - refreeze
        excess = torch.clamp(meltwater - par["parSSC"] * snowpack, min=0.0)
        meltwater = meltwater - excess
        return snowpack, meltwater, excess, melt, refreeze, old_snow

    def _snow(self, states, temp, temp_glac, rain_land, snow_land, rain_glac, snow_glac, par):
        land_fraction = 1.0 - states["glac_frac"] if self.use_glacier else torch.ones_like(states["snow_land"])
        land_depth = states["snow_land"] / land_fraction.clamp_min(self.eps)
        land_cover = torch.where(
            land_fraction > 0.0, self._snow_cover_fraction(land_depth),
            torch.zeros_like(land_depth),
        )
        rain_land_snow = rain_land * land_cover
        rain_land_soil = rain_land * (1.0 - land_cover)
        states["snow_land"], states["snow_liq_land"], land_out, land_melt, land_refreeze, _ = self._snowpack_step(
            states["snow_land"], states["snow_liq_land"], temp, rain_land_snow, snow_land, par,
            area_fraction=land_fraction,
        )

        if self.use_glacier:
            glac_cover = self._snow_cover_fraction(states["snow_glac"])
            rain_glac_snow = rain_glac * glac_cover
            rain_glac_ice = rain_glac * (1.0 - glac_cover)
            states["snow_glac"], states["snow_liq_glac"], glac_out, glac_melt, glac_refreeze, old_snow_glac = self._snowpack_step(
                states["snow_glac"], states["snow_liq_glac"], temp_glac, rain_glac_snow, snow_glac, par
            )
            glac_frac = states["glac_frac"]
            glac_out_area = glac_out * glac_frac
            glac_melt_area = glac_melt * glac_frac
            glac_refreeze_area = glac_refreeze * glac_frac
            rain_glac_ice_area = rain_glac_ice * glac_frac
        else:
            states["snow_glac"] = torch.zeros_like(states["snow_glac"])
            states["snow_liq_glac"] = torch.zeros_like(states["snow_liq_glac"])
            glac_out_area = torch.zeros_like(land_out)
            glac_melt_area = torch.zeros_like(land_melt)
            glac_refreeze_area = torch.zeros_like(land_refreeze)
            old_snow_glac = torch.zeros_like(land_out)
            rain_glac_ice_area = torch.zeros_like(land_out)

        states["snow"] = states["snow_land"] + states["snow_glac"] * states["glac_frac"]
        states["snow_liq"] = states["snow_liq_land"] + states["snow_liq_glac"] * states["glac_frac"]
        # Land snow-store outflow feeds SM; glacier snow-store outflow bypasses SM.
        snow_recharge = land_out
        return (
            states, snow_recharge, land_melt + glac_melt_area, land_refreeze + glac_refreeze_area,
            rain_land_soil, rain_glac_ice_area, old_snow_glac, glac_out_area,
            land_melt, glac_melt_area, land_refreeze, glac_refreeze_area,
        )

    def _glacier(self, states, glac_temp, rain_glac_ice, old_snow_glac, attrs, par, glac_frac):
        old_depth = old_snow_glac
        new_depth = states["snow_glac"]
        old_cover = self._snow_cover_fraction(old_depth)
        new_cover = self._snow_cover_fraction(new_depth)
        glac_heat = glac_temp - par["parTCritGlac"]
        partial_heat = torch.clamp(glac_heat - old_depth / (par["parDDFS"] + self.eps), min=0.0) * old_cover * (1.0 - new_cover)
        full_heat = torch.clamp(glac_heat, min=0.0) * (1.0 - old_cover) * (1.0 - new_cover)
        melt = (partial_heat + full_heat) * par["parDDFG"] * glac_frac
        sub_area = attrs["sub_area"].unsqueeze(-1)
        available_melt = torch.clamp(states["glac_storage"] / sub_area * 1000.0 - states["acc_glac_melt"], min=0.0)
        melt = torch.minimum(melt, available_melt)
        states["acc_glac_melt"] = states["acc_glac_melt"] + melt
        return states, melt, rain_glac_ice

    def _freeze_thaw(self, states, temp, attrs, par):
        # Seasonal Stefan approximation: one-way degree days and continuous fronts.
        zero = torch.zeros_like(temp)
        # Thermal depth is independent of the shallower HBV storage column.
        # Soil/groundwater mobility below still uses the actual hydrological depths.
        limit = torch.full_like(temp, self.thermal_column_depth_mm)
        if "season_freeze_max" not in states:
            states["thermal_mean"] = temp.clone()
            states["thermal_season"] = zero
            states["season_days"] = zero
            states["cold_index"] = zero
            states["warm_index"] = zero
            states["season_freeze_max"] = zero
            states["residual_frozen_thickness"] = zero
            states["frozen_thickness"] = zero
        mean = states["thermal_mean"] + (temp - states["thermal_mean"]) / 7.0
        anomaly = mean - par["parTCritSoil"]
        candidate = torch.where(anomaly < -0.5, -torch.ones_like(zero),
                             torch.where(anomaly > 0.5, torch.ones_like(zero), states["thermal_season"]))
        # Count completed days in the current season; initial classification is free.
        can_switch = (states["thermal_season"] == 0) | (states["season_days"] >= self.min_season_days)
        season = torch.where(can_switch, candidate, states["thermal_season"])
        changed = season != states["thermal_season"]
        states["season_days"] = torch.where(
            changed, torch.ones_like(zero),
            torch.where(season != 0, states["season_days"] + 1.0, zero),
        )
        start_freeze = changed & (season < 0)
        start_thaw = changed & (season > 0)
        states["residual_frozen_thickness"] = torch.where(
            start_freeze, states["frozen_thickness"], states["residual_frozen_thickness"]
        )
        cold = torch.clamp(par["parTCritSoil"] - temp, min=0.0)
        warm = torch.clamp(temp - par["parTCritSoil"], min=0.0)
        # Stefan coefficient following Gao et al. (2022):
        # z = sqrt(2 * k * degree_seconds / (L * omega * rho)).
        thermal_conductivity = float(self.config.get("stefan_thermal_conductivity", 2.0))
        latent_heat = float(self.config.get("stefan_latent_heat", 3.35e5))
        soil_water_content = float(self.config.get("stefan_soil_water_content", 0.12))
        soil_bulk_density = float(self.config.get("stefan_soil_bulk_density", 1000.0))
        coefficient = (
            2.0 * thermal_conductivity
            / (latent_heat * soil_water_content * soil_bulk_density)
        )
        n_thaw = float(self.config.get("stefan_n_thaw", 1.0))
        cold_increment = torch.clamp(par["parN"], 0.05, 1.0) * cold
        warm_increment = n_thaw * warm
        c = coefficient * 86400.0
        if self.use_freeze_heterogeneity and self.nmul > 1:
            slope = attrs["slope"].unsqueeze(-1)
            slope = torch.clamp(torch.where(slope > 1.0, slope / 30.0, slope / 0.30), 0.20, 2.0)
            scale = torch.exp(self.freeze_heterogeneity_strength * slope * self.freeze_heterogeneity_q)
            c = c * scale.square()
        def index_from_depth(depth):
            metres = depth / 1000.0
            return (metres.square() + 2e-6 * metres) / c
        # Carry incomplete thaw as equivalent frozen thickness, not a spatial
        # multilayer reconstruction. This preserves thickness but loses position.
        initial_fi = index_from_depth(states["residual_frozen_thickness"])
        fi = torch.where(start_freeze, initial_fi, states["cold_index"])
        fi = torch.where(season < 0, fi + cold_increment, zero)
        states["cold_index"] = torch.minimum(fi, index_from_depth(limit))
        freeze_front = (torch.sqrt(c * states["cold_index"] + 1e-12) - 1e-6) * 1000.0
        maximum = torch.where(start_freeze, zero, states["season_freeze_max"])
        maximum = torch.where(season < 0, torch.maximum(maximum, freeze_front), maximum)
        states["season_freeze_max"] = maximum
        # Preserve remaining thickness across the switch before today's energy.
        initial_thaw_depth = torch.clamp(maximum - states["frozen_thickness"], min=0.0)
        ti = torch.where(start_thaw, index_from_depth(initial_thaw_depth), states["warm_index"])
        ti = torch.where(season > 0, ti + warm_increment, zero)
        states["warm_index"] = torch.minimum(ti, index_from_depth(maximum))
        thaw_front = (torch.sqrt(c * states["warm_index"] + 1e-12) - 1e-6) * 1000.0
        bottom = torch.where(season < 0, freeze_front, maximum)
        top = torch.minimum(thaw_front, bottom)
        states["thermal_mean"], states["thermal_season"] = mean, season
        states["frozen_thickness"] = torch.clamp(bottom - top, min=0.0)
        return states, top, bottom

    def _frozen_overlap(self, lower, upper, z_thaw, z_freeze):
        return torch.clamp(torch.minimum(z_freeze, upper) - torch.maximum(z_thaw, lower), min=0.0)

    def _soil_groundwater(self, states, rain, snow_recharge, glacier_recharge, pet, z_thaw, z_freeze, attrs, par):
        root_cap = torch.clamp(par["parFC"], min=1.0)
        root_depth = torch.clamp(attrs["root_depth"].unsqueeze(-1), min=1.0)
        sub_depth = torch.clamp(attrs["sub_depth"].unsqueeze(-1), min=1.0)
        soil_depth = root_depth + sub_depth
        thermal_depth = torch.maximum(
            torch.full_like(soil_depth, self.thermal_column_depth_mm),
            soil_depth + self.eps,
        )
        zero_depth = torch.zeros_like(root_depth)

        frozen_root_depth = self._frozen_overlap(zero_depth, root_depth, z_thaw, z_freeze)
        frozen_uz_depth = self._frozen_overlap(zero_depth, soil_depth, z_thaw, z_freeze)
        frozen_lz_depth = self._frozen_overlap(zero_depth, thermal_depth, z_thaw, z_freeze)
        sm_freeze_frac = torch.clamp(
            frozen_root_depth / (root_depth + self.eps), min=0.0, max=1.0
        )
        uz_freeze_frac = torch.clamp(
            frozen_uz_depth / (soil_depth + self.eps), min=0.0, max=1.0
        )
        lz_freeze_frac = torch.clamp(
            frozen_lz_depth / (thermal_depth + self.eps),
            min=0.0,
            max=1.0,
        )
        thawed_root_frac = torch.clamp(1.0 - sm_freeze_frac, min=self.eps, max=1.0)
        thawed_uz_frac = torch.clamp(1.0 - uz_freeze_frac, min=self.eps, max=1.0)
        thawed_lz_frac = torch.clamp(1.0 - lz_freeze_frac, min=self.eps, max=1.0)
        sm_mobile_frac = 0.1 + 0.9 * torch.pow(thawed_root_frac, par["parBFS"])
        uz_mobile_frac = 0.1 + 0.9 * torch.pow(thawed_uz_frac, par["parBFUZ"])
        lz_mobile_frac = 0.1 + 0.9 * torch.pow(thawed_lz_frac, par["parBFLZ"])

        water_input = rain + snow_recharge
        infil = water_input

        root_mobile = states["root"] * sm_mobile_frac
        soil_wetness = torch.clamp(root_mobile / root_cap, min=0.0, max=1.0).pow(par["parBETA"])
        # Partition today's input using antecedent moisture, then limit available water.
        states["root"] = states["root"] + infil
        root_mobile = states["root"] * sm_mobile_frac
        root_perc = torch.minimum(infil * soil_wetness, root_mobile)
        states["root"] = states["root"] - root_perc

        root_excess = torch.clamp(states["root"] - root_cap, min=0.0)
        states["root"] = states["root"] - root_excess

        root_mobile = states["root"] * sm_mobile_frac
        evap_factor = torch.clamp(root_mobile / (par["parLP"] * root_cap + self.eps), min=1.0e-6, max=1.0)
        evap_factor = evap_factor.pow(par["parBETAET"])
        aet = torch.minimum(pet * evap_factor, root_mobile)
        states["root"] = torch.clamp(states["root"] - aet, min=self.eps)

        gw_mobile = states["gw"] * lz_mobile_frac
        root_space = torch.clamp(root_cap - states["root"], min=0.0)
        caprise = torch.minimum(
            torch.minimum(gw_mobile, root_space),
            par["parC"] * gw_mobile * (1.0 - torch.clamp(states["root"] / (root_cap + self.eps), 0.0, 1.0)),
        )
        states["root"] = torch.clamp(states["root"] + caprise, min=self.eps)
        states["gw"] = torch.clamp(states["gw"] - caprise, min=self.eps)

        # Soil-capacity excess enters the HBV response reservoir, not direct runoff.
        states["sub"] = states["sub"] + root_perc + root_excess + glacier_recharge
        sub_mobile = states["sub"] * uz_mobile_frac
        sub_perc = torch.minimum(par["parPERC"], sub_mobile)
        states["sub"] = states["sub"] - sub_perc
        sub_mobile = states["sub"] * uz_mobile_frac
        Q0 = par["parK0"] * torch.clamp(sub_mobile - par["parUZL"], min=0.0)
        Q0 = torch.minimum(Q0, states["sub"])
        states["sub"] = states["sub"] - Q0

        sub_mobile = states["sub"] * uz_mobile_frac
        Q1 = par["parK1"] * sub_mobile
        Q1 = torch.minimum(Q1, states["sub"])
        states["sub"] = states["sub"] - Q1

        states["gw"] = states["gw"] + sub_perc
        gw_mobile = states["gw"] * lz_mobile_frac
        Q2 = par["parK2"] * gw_mobile
        Q2 = torch.minimum(Q2, states["gw"])
        states["gw"] = states["gw"] - Q2

        states["gw_recharge"] = sub_perc
        states["baseflow"] = Q2
        surfaceflow = Q0
        uz_above_threshold = torch.clamp(states["sub"] * uz_mobile_frac - par["parUZL"], min=0.0)
        return (
            states, surfaceflow, Q1, Q2, root_perc, sub_perc,
            aet, infil,
            sm_mobile_frac, uz_mobile_frac, lz_mobile_frac,
            root_excess, uz_above_threshold, caprise,
            water_input, soil_wetness, evap_factor, Q0, Q1, Q2,
            sm_freeze_frac, uz_freeze_frac, lz_freeze_frac,
        )

    def _member_mean(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() > 1 and value.shape[-1] == self.nmul:
            return value.mean(dim=-1)
        return value

    def _sphy_retention(self, par):
        """Daily dimensionless memory coefficient, independent of substeps."""
        return par["parK"].clamp(0.0, self.sphy_k_max)

    def _sphy_routing_sequence(self, states, runoff, attrs, state_name, retention):
        """Accumulate UNFILTERED upstream runoff, then smooth once at each outlet.

        runoff: [day, basin] or [day, component, basin], in mm/day.
        State is m3/s even when the requested output is mm/day. This matches
        SPHY basic routing, not a cascade of reach reservoirs.
        """
        if runoff.shape[0] == 0:
            return states, runoff.clone()
        accumulated = runoff * attrs["sub_area"].clamp_min(self.eps) / (1000.0 * 86400.0)
        for idx_name, _, down_name, valid_idx_name in self.route_level_names:
            down = getattr(self, down_name)
            if down.numel():
                idx = getattr(self, idx_name).index_select(0, getattr(self, valid_idx_name))
                # Pass pre-filter accumulation downstream, never routed outflow.
                accumulated = accumulated.index_add(-1, down, accumulated.index_select(-1, idx))
        forcing = (1.0 - retention) * accumulated
        result = torch.cat(((forcing[0] + retention * states[state_name]).unsqueeze(0), forcing[1:]), dim=0)
        stride, power = 1, retention
        while stride < result.shape[0]:
            result = torch.cat((result[:stride], result[stride:] + power * result[:-stride]), dim=0)
            power = power * power
            stride *= 2
        states[state_name] = result[-1]
        states[f"{state_name}_in"] = accumulated[-1]
        states["route_in"] = accumulated[-1].sum(0) if runoff.dim() == 3 else accumulated[-1]
        if self.output_unit not in {"m3/s", "cms", "m3s"}:
            result = result / self.route_area_tensor.to(result).clamp_min(self.eps) * 1000.0 * 86400.0
        return states, result

    def _routing_effective_parameters(self, par):
        if self.routing_method == "sphy":
            raise ValueError("SPHY has a daily retention coefficient; use _sphy_retention instead")
        dt = 1.0 / self.routing_substeps
        x = torch.clamp(par["parX"], self.eps, self.routing_x_max)
        k_min = dt / (2.0 * (1.0 - x))
        k_max = torch.clamp(dt / (2.0 * x), max=self.routing_k_cap_hours / 24.0)
        k = k_min + torch.clamp(par["parK"], 0.0, 1.0) * (k_max - k_min)
        return x, k

    def _channel_routing_coefficients(self, par) -> dict[str, Any]:
        if self.routing_method == "sphy":
            return {"sphy_k": self._sphy_retention(par)}
        coeffs = {}
        dt = 1.0 / self.routing_substeps
        effective_x, effective_k = self._routing_effective_parameters(par)
        for idx_name, _, _, _ in self.route_level_names:
            idx = getattr(self, idx_name)
            x = effective_x.index_select(0, idx)
            k = effective_k.index_select(0, idx)
            denom = 2.0 * k * (1.0 - x) + dt
            coeffs[idx_name] = (
                (dt - 2.0 * k * x) / denom,
                (dt + 2.0 * k * x) / denom,
                (2.0 * k * (1.0 - x) - dt) / denom,
            )
        return coeffs

    def _channel_routing(self, states, runoff, attrs, state_name: str, routing_coeffs):
        # Daily runoff is a constant rate over the day. Keep endpoint states for
        # the next substep/day, but return the mean of substep discharge outputs.
        if self.routing_method == "sphy":
            states, daily = self._sphy_routing_sequence(states, runoff.unsqueeze(0), attrs, state_name, routing_coeffs["sphy_k"])
            return states, daily[0]
        daily_q = torch.zeros_like(runoff)
        for _ in range(self.routing_substeps):
            states, q = self._channel_routing_step(
                states, runoff, attrs, state_name, routing_coeffs
            )
            daily_q = daily_q + q
        return states, daily_q / self.routing_substeps

    def _channel_routing_sequence(self, states, runoff, attrs, state_name, routing_coeffs):
        """Route [day, (component), basin] using a parallel affine prefix scan."""
        if self.routing_method == "sphy":
            return self._sphy_routing_sequence(states, runoff, attrs, state_name, routing_coeffs["sphy_k"])
        days = runoff.shape[0]
        local = (runoff * attrs["sub_area"].clamp_min(self.eps) / 1000.0 / 86400.0)
        local = local.repeat_interleave(self.routing_substeps, dim=0)
        upstream = torch.zeros_like(local)
        ins, outs = [], []
        for idx_name, _, down_name, valid_idx_name in self.route_level_names:
            idx = getattr(self, idx_name)
            inflow = local.index_select(-1, idx) + upstream.index_select(-1, idx)
            previous_in = states[f"{state_name}_in"].index_select(-1, idx)
            previous_out = states[state_name].index_select(-1, idx)
            c1, c2, c3 = routing_coeffs[idx_name]
            delayed = torch.cat((previous_in.unsqueeze(0), inflow[:-1]), dim=0)
            forcing = c1 * inflow + c2 * delayed
            result = torch.cat(((forcing[0] + c3 * previous_out).unsqueeze(0), forcing[1:]), dim=0)
            stride = 1
            power = c3
            # Doubling composes y[t] = a*y[t-1] + b[t] in log2(T) stages.
            while stride < result.shape[0]:
                result = torch.cat((result[:stride], result[stride:] + power * result[:-stride]), dim=0)
                power = power * power
                stride *= 2
            down = getattr(self, down_name)
            if down.numel():
                valid = getattr(self, valid_idx_name)
                upstream = upstream.index_add(-1, down, result.index_select(-1, valid))
            ins.append(inflow[-1])
            outs.append(result)
        flow = torch.zeros_like(local).index_copy(-1, self.route_all_indices, torch.cat(outs, dim=-1))
        final_in = torch.zeros_like(states[f"{state_name}_in"]).index_copy(
            -1, self.route_all_indices, torch.cat(ins, dim=-1)
        )
        states[f"{state_name}_in"] = final_in
        states[state_name] = flow[-1]
        states["route_in"] = final_in.sum(0) if runoff.dim() == 3 else final_in
        daily = flow.reshape(days, self.routing_substeps, *runoff.shape[1:]).mean(1)
        if self.output_unit not in {"m3/s", "cms", "m3s"}:
            daily = daily / self.route_area_tensor.to(daily).clamp_min(self.eps) * 1000.0 * 86400.0
        return states, daily

    def _channel_routing_step(self, states, runoff, attrs, state_name: str, routing_coeffs):
        if self.routing_method == "sphy":
            return self._channel_routing(states, runoff, attrs, state_name, routing_coeffs)
        is_components = runoff.dim() == 2
        sub_area = torch.clamp(attrs["sub_area"], min=self.eps)
        local_q = runoff * (sub_area.unsqueeze(0) if is_components else sub_area) / 1000.0 / 86400.0

        dim = 1 if is_components else 0
        upstream = torch.zeros_like(local_q)
        q_in_levels, q_out_levels = [], []
        old_in_name = f"{state_name}_in"

        for idx_name, _, down_name, valid_idx_name in self.route_level_names:
            idx = getattr(self, idx_name)
            down_valid = getattr(self, down_name)
            valid_idx = getattr(self, valid_idx_name)
            lateral = local_q.index_select(dim, idx)
            q_in = upstream.index_select(dim, idx) + lateral
            old_in = states[old_in_name].index_select(dim, idx)
            old_q = states[state_name].index_select(dim, idx)
            c1, c2, c3 = routing_coeffs[idx_name]
            if is_components:
                c1 = c1.unsqueeze(0)
                c2 = c2.unsqueeze(0)
                c3 = c3.unsqueeze(0)
            routed = c1 * q_in + c2 * old_in + c3 * old_q
            q_out_level = torch.clamp(routed, min=0.0)

            if down_valid.numel() > 0:
                upstream = upstream.index_add(dim, down_valid, q_out_level.index_select(dim, valid_idx))

            q_in_levels.append(q_in)
            q_out_levels.append(q_out_level)

        if is_components:
            q_in = torch.zeros_like(local_q).index_copy(1, self.route_all_indices, torch.cat(q_in_levels, dim=1))
            q_out = torch.zeros_like(local_q).index_copy(1, self.route_all_indices, torch.cat(q_out_levels, dim=1))
            states["route_in"] = q_in.sum(dim=0)
            states[old_in_name] = q_in
            states[state_name] = q_out
            if self.output_unit in {"m3/s", "cms", "m3s"}:
                return states, q_out
            route_area = torch.clamp(self.route_area_tensor.to(dtype=runoff.dtype), min=self.eps)
            return states, q_out / route_area.unsqueeze(0) * 1000.0 * 86400.0

        q_in = torch.zeros_like(local_q).scatter(0, self.route_all_indices, torch.cat(q_in_levels))
        q_out = torch.zeros_like(local_q).scatter(0, self.route_all_indices, torch.cat(q_out_levels))
        states["route_in"] = q_in
        states[old_in_name] = q_in
        states[state_name] = q_out
        states["q"] = q_out
        if self.output_unit in {"m3/s", "cms", "m3s"}:
            return states, q_out
        route_area = torch.clamp(self.route_area_tensor.to(dtype=runoff.dtype), min=self.eps)
        return states, q_out / route_area * 1000.0 * 86400.0

    def _debug_check(self, t: int, stage: str, tensors: dict[str, torch.Tensor], states: dict[str, torch.Tensor] | None = None) -> None:
        if not self.debug_nan:
            return

        check_items = dict(tensors)
        if states is not None:
            check_items.update({f"state.{name}": value for name, value in states.items()})

        for name, value in check_items.items():
            if not torch.is_tensor(value):
                continue
            finite = torch.isfinite(value)
            if finite.all():
                continue

            bad = torch.nonzero(~finite.reshape(-1), as_tuple=False).flatten()
            sample = bad[:10].detach().cpu().tolist()
            finite_values = value[finite]
            if finite_values.numel() > 0:
                vmin = float(finite_values.min().detach().cpu())
                vmax = float(finite_values.max().detach().cpu())
            else:
                vmin = float("nan")
                vmax = float("nan")
            raise RuntimeError(
                f"Non-finite tensor in CryoHydro: t={t}, stage={stage}, name={name}, "
                f"bad={bad.numel()}/{value.numel()}, sample_flat_indices={sample}, "
                f"finite_min={vmin:.6g}, finite_max={vmax:.6g}"
            )

    def _snow_cover_fraction(self, snowpack: torch.Tensor) -> torch.Tensor:
        # Remove the sigmoid offset so a snow-free surface intercepts no rain.
        baseline = self._soft_gate(torch.full_like(snowpack, -1.0), tau=2.0)
        cover = self._soft_gate(snowpack - 1.0, tau=2.0)
        return torch.clamp((cover - baseline) / (1.0 - baseline), min=0.0, max=1.0)

    def _soft_gate(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return torch.sigmoid(x / (tau + self.eps))
