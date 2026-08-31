"""
Module partagé - bellman_config.py
=====================================

CONFIGURATION A RENSEIGNEE EN BAS DU FICHIER

Lit tous les paramètres depuis des entités input_number Home Assistant.
Est ensuite importé par les 3 apps AppDaemon.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BellmanConfig:

    # Parametres numeriques (lus depuis input_number HA)
    D_segments:      float
    N_jours_horizon: int
    n_personnes:     int
    conso_perso_ecs: float
    E_min:           float
    E_max:           float
    step_stock:      float
    P_nom_B:         float
    P_nom_P:         float
    D_tempo_pompe:   float
    D_tempo_ballon:  float
    lambda_solaire:  float
    lambda_reseau:   float
    E_pertes_min_dt: float
    E_pertes_max_dt: float

    # Chemins et credentials
    data_dir:    str
    cons_path:   str
    resource_id: str
    api_key:     str

    # Noms des entites HA
    entity_temp_piscine:        str
    entity_crepuscule:          str
    entity_ballon_energie:      str
    entity_energie_pompe:       str
    entity_compteur_k_pompe:    str
    entity_compteur_k_ballon:   str
    entity_switch_ballon:       str
    entity_mode_auto_ballon:    str
    entity_puissance_ballon:    str
    entity_switch_pompe:        str
    entity_mode_auto_piscine:   str

    # ── Proprietes calculees ───────────────────────────────────────────────
    @property
    def N_segments(self) -> int:
        return int(0.5 / self.D_segments)

    @property
    def K_max_pompe(self) -> int:
        return round(self.D_tempo_pompe / self.D_segments)

    @property
    def K_max_ballon(self) -> int:
        return round(self.D_tempo_ballon / self.D_segments)

    @property
    def alpha_pertes(self) -> float:
        return (self.E_pertes_max_dt - self.E_pertes_min_dt) / (self.E_max - self.E_min)

    @property
    def S_vector(self) -> np.ndarray:
        return np.arange(self.E_min, self.E_max + self.step_stock, self.step_stock)

    @property
    def K_vector_pompe(self) -> np.ndarray:
        tmp = np.arange(-self.K_max_pompe, self.K_max_pompe + 1, 1)
        return tmp[tmp != 0]  # On retire le 0 qui est inutile

    @property
    def K_vector_ballon(self) -> np.ndarray:
        tmp = np.arange(-self.K_max_ballon, self.K_max_ballon + 1, 1)
        return tmp[tmp != 0] # On retire le 0 qui est inutile

    @property
    def N_stock(self) -> int:
        return len(self.S_vector)

    @property
    def N_K_pompe(self) -> int:
        return len(self.K_vector_pompe)

    @property
    def N_K_ballon(self) -> int:
        return len(self.K_vector_ballon)

    # ── Constructeur ──────────────────────────────────────────────────────
    @classmethod
    def from_ha(cls, app) -> "BellmanConfig":
        """Charge la config depuis les input_number HA."""

        def ha(entity_id: str) -> float:
            raw = app.get_state(entity_id)
            if raw in (None, "unavailable", "unknown"):
                raise ValueError(f"Entite HA indisponible : {entity_id}")
            return float(raw)

        D_seg = ha("input_number.bellman_d_segments")/60
        N_seg = int(0.5/D_seg)

        return cls(
            D_segments      = D_seg,
            N_jours_horizon = int(ha("input_number.bellman_n_jours_horizon")),
            n_personnes     = int(ha("input_number.bellman_n_personnes")),
            conso_perso_ecs = ha("input_number.bellman_conso_par_personne"),
            E_min           = ha("input_number.bellman_e_min"),
            E_max           = ha("input_number.bellman_e_max"),
            step_stock      = ha("input_number.bellman_step_stock"),
            P_nom_B         = 2.200,
            P_nom_P         = 0.350,
            D_tempo_pompe   = ha("input_number.bellman_d_tempo_pompe")/60,
            D_tempo_ballon  = ha("input_number.bellman_d_tempo_ballon")/60,
            lambda_solaire  = ha("input_number.bellman_lambda_solaire"),
            lambda_reseau   = ha("input_number.bellman_lambda_reseau"),
            E_pertes_min_dt = ha("input_number.bellman_e_pertes_min") / 48 / N_seg,
            E_pertes_max_dt = ha("input_number.bellman_e_pertes_max") / 48 / N_seg,
            data_dir        = "/config/apps/automatisation_bellman/energy_data/",
            cons_path       = "/config/apps/automatisation_bellman/energy_data/profil_consommation.csv",
            resource_id     = "",
            api_key         = "",
            entity_temp_piscine         = "sensor.moyenne_temperature_piscine",
            entity_crepuscule           = "sensor.sun_next_dusk",
            entity_ballon_energie       = "input_number.ballon_ecs_energie_kwh",
            entity_energie_pompe        = "sensor.compteur_piscine",
            entity_compteur_k_pompe     = "sensor.pompe_piscine_compteur_k",
            entity_compteur_k_ballon    = "sensor.ballon_compteur_k",
            entity_switch_ballon        = "switch.buandrie_prise_chauffe_eau",
            entity_mode_auto_ballon     = "input_select.mode_fonctionnement_chauffe_eau",
            entity_puissance_ballon     = "sensor.buandrie_prise_chauffe_eau_puissance",
            entity_switch_pompe         = "switch.piscine_pompe",
            entity_mode_auto_piscine    = "input_select.mode_de_fonctionnement_piscine",
        )

    def log_summary(self, app) -> None:
        app.log(
            f"BellmanConfig : N_seg={self.N_segments} D_seg={self.D_segments}h "
            f"K_max_pompe={self.K_max_pompe} K_max_ballon={self.K_max_ballon} | "
            f"E=[{self.E_min},{self.E_max}]kWh "
            f"P_B={self.P_nom_B}kW P_P={self.P_nom_P}kW | "
            f"tarifs sol={self.lambda_solaire} res={self.lambda_reseau} EUR/kWh"
        )