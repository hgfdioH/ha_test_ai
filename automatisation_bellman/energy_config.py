"""
AppDaemon App - energy_config.py
=================================
Charge la configuration via BellmanConfig, recupère les prévisions solaires
et construit les vecteurs d'état. Déclenche 'energy_config_ready'.
"""

import hassapi as hass
import numpy as np
import pandas as pd
import os

from automatisation_bellman.bellman_config import BellmanConfig


class EnergyConfig(hass.Hass):

    def initialize(self):
        self._set_status("En attente")
        self.run_in(self._compute_config, 5)
        self.run_daily(self._compute_config, "00:05:00")
        self.log("EnergyConfig initialise.")

    # ------------------------------------------------------------------

    def _compute_config(self, kwargs):
        self.log("EnergyConfig: debut de la configuration...")
        self._set_status("Calculs en cours")

        # ── Chargement de la config partage ──────────────────────────
        cfg = BellmanConfig.from_ha(self)
        cfg.log_summary(self)
        os.makedirs(cfg.data_dir, exist_ok=True)

        # ── Valeurs dynamiques depuis HA ───────────────────────────────
        T_eau       = self._read_float(cfg.entity_temp_piscine, default=24.0)
        self.log(f"  T_eau={T_eau}C kWh")

        D_piscine_min = T_eau / 2
        D_piscine_min_grid = D_piscine_min//cfg.D_segments*cfg.D_segments
        D_vector      = np.arange(0, D_piscine_min_grid + cfg.D_segments, cfg.D_segments)

        # ── Profil de consommation ─────────────────────────────────────
        try:
            df_cons  = pd.read_csv(cfg.cons_path)
            C_elec_h = df_cons["C_elec_kWh"].to_numpy()
            C_ecs_h  = df_cons["C_ecs_kWh"].to_numpy()
        except FileNotFoundError:
            self.log(f"EnergyConfig: fichier introuvable : {cfg.cons_path}", level="ERROR")
            self._set_status("Erreur")
            return

        # ── Previsions solaires ────────────────────────────────────────
        Q_th = self._fetch_solar(cfg)
        if Q_th is None:
            self._set_status("Erreur")
            return

        T_days = Q_th.shape[0]
        J      = T_days * 48 * cfg.N_segments

        # ── Interpolation lineaire demi-heure -> segment fin ─────────────
        # Les previsions/profils sont fournis par pas de 30 min. On les
        # interpole linairement entre les points connus (plutot que de
        # repliquer platement la valeur de la demi-heure sur chaque
        # segment) pour que N_segments soit reellement discriminant.
        # C_elec_h et C_ecs_h sont un profil "type" d'une seule journee
        # (48 valeurs) : on le repete sur T_days avant interpolation, ce
        # qui referme aussi naturellement la boucle entre la derniere
        # demi-heure d'un jour et la premiere du suivant (meme valeur).
        Q_fine      = self._interp_to_segments(Q_th.flatten(),            T_days, cfg.N_segments)
        C_elec_fine = self._interp_to_segments(np.tile(C_elec_h, T_days), T_days, cfg.N_segments)
        C_ecs_fine  = self._interp_to_segments(np.tile(C_ecs_h,  T_days), T_days, cfg.N_segments)

        # ── Aplatissement (t, h) -> j ───────────────────────────────────
        surplus_j        = np.zeros(J + 1)
        ecs_j            = np.zeros(J + 1)
        journee_terminee = np.zeros(J + 1, dtype=bool)

        surplus_j[1:] = np.maximum(Q_fine - C_elec_fine / 0.5, 0.0)  # /0.5 car conversion en kW moyen
        ecs_j[1:]     = C_ecs_fine * cfg.conso_perso_ecs / 2.5

        pas_par_jour = 48 * cfg.N_segments
        journee_terminee[pas_par_jour::pas_par_jour] = True

        # ── Sauvegarde des arrays ──────────────────────────────────────
        d = cfg.data_dir
        np.save(d + "S_vector",                 cfg.S_vector)
        np.save(d + "D_vector",                 D_vector)
        np.save(d + "K_vector_pompe",                 cfg.K_vector_pompe)
        np.save(d + "K_vector_ballon",          cfg.K_vector_ballon)
        np.save(d + "surplus_j_vector",         surplus_j)
        np.save(d + "ecs_j_vector",             ecs_j)
        np.save(d + "journee_terminee_vector",  journee_terminee)
        np.save(d + "C_ecs_h",                  C_ecs_h)

        self.log(f"EnergyConfig: prete | T_days={T_days}, J={J}")
        self._set_status("Prêt")

        # ── Declenchement de PolicyCreator ────────────────────────────
        self.fire_event(
            "energy_config_ready",
            J=J,
        )

    # ------------------------------------------------------------------

    def _interp_to_segments(self, values_par_demi_heure, T_days, N_segments):
        """Interpole linairement une serie fournie par demi-heure (longueur
        T_days*48) vers une serie fine a N_segments segments par demi-heure.

        Chaque valeur source est ancree au DEBUT de sa demi-heure. Pour un
        segment situe a une fraction f (0 <= f < 1) a l'interieur d'une
        demi-heure, la valeur interpolee est :
            valeur[h] + f * (valeur[h+1] - valeur[h])
        Le dernier segment du dernier jour n'a pas de "valeur suivante" :
        il reste a la derniere valeur connue (pas d'extrapolation).

        Exemple (N_segments=4) : prod 9h30-10h=2, prod 10h-10h30=4
        -> segments 9h30/9h37/9h45/9h52 = 2.0 / 2.5 / 3.0 / 3.5, puis 4.0
        pile au segment de 10h00 (premiere valeur de la demi-heure suivante).
        """
        N          = len(values_par_demi_heure)
        t_coarse   = np.arange(N) * 0.5                       # heures, pas 30 min
        D_segments = 0.5 / N_segments
        J          = T_days * 48 * N_segments
        t_fine     = np.arange(J) * D_segments                 # heures, pas fin
        return np.interp(t_fine, t_coarse, values_par_demi_heure)

    # ------------------------------------------------------------------

    def _fetch_solar(self, cfg: BellmanConfig):
        CSV_PATH  = cfg.data_dir + "GetRooftopSiteForecast.csv"
        today_str = pd.Timestamp.now(tz="Europe/Paris").strftime("%Y-%m-%d")

        cache_valid = False
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                cache_valid = f.readline().strip() == today_str
        except FileNotFoundError:
            pass

        if cache_valid:
            self.log("[solar] Cache valide, lecture locale.")
            df = pd.read_csv(CSV_PATH, skiprows=1)
        else:
            self.log("[solar] Telechargement en cours...")
            url = (
                f"https://api.solcast.com.au/rooftop_sites/{cfg.resource_id}/forecasts"
                f"?format=csv&hours=336&api_key={cfg.api_key}"
            )
            try:
                df = pd.read_csv(url)
                with open(CSV_PATH, "w", encoding="utf-8") as f:
                    f.write(today_str + "\n")
                    df.to_csv(f, index=False)
                self.log(f"[solar] Sauvegarde : {CSV_PATH}")
            except Exception as exc:
                self.log(f"[solar] Erreur : {exc}", level="ERROR")
                return None

        df["PeriodEnd"] = (
            pd.to_datetime(df["PeriodEnd"], format="mixed", utc=True)
            .dt.tz_convert("Europe/Paris")
        )
        today        = pd.Timestamp.now(tz="Europe/Paris").normalize()
        period_start = df["PeriodEnd"] - pd.Timedelta(minutes=30)
        df["t"]      = (period_start.dt.normalize() - today).dt.days
        df["h"]      = period_start.dt.hour * 2 + period_start.dt.minute // 30

        df = df.loc[
            (df["t"] >= 0) & (df["t"] < cfg.N_jours_horizon)
            & (df["h"] >= 0) & (df["h"] <= 47)
        ]
        T_days = int(df["t"].max()) + 1
        Q      = np.zeros((T_days, 48))
        Q[df["t"].to_numpy(dtype=int), df["h"].to_numpy(dtype=int)] = df["PvEstimate"].to_numpy()
        return Q

    # ------------------------------------------------------------------

    def _read_float(self, entity_id, default):
        raw = self.get_state(entity_id)
        if raw in (None, "unavailable", "unknown"):
            self.log(f"EnergyConfig: {entity_id} indisponible -> defaut={default}", level="WARNING")
            return default
        return float(raw)

    def _set_status(self, status):
        self.set_state("sensor.energy_opt_config_status", state=status,
                        attributes={"friendly_name": "Statut config energie"})