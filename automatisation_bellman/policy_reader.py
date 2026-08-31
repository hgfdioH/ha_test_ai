"""
AppDaemon App - policy_reader.py
==================================
Toutes les D_segments heures, lit l'état (S, D, K) depuis HA,
consulte la politique optimale et commande le chauffe-eau et la pompe.
"""

import hassapi as hass
import numpy as np
import pandas as pd
import datetime

from automatisation_bellman.bellman_config import BellmanConfig


class PolicyReader(hass.Hass):

    def initialize(self):
        cfg = BellmanConfig.from_ha(self)

        self.pol_uB           = None
        self.pol_uP           = None
        self.S_vector         = None
        self.D_vector         = None
        self.K_vector_pompe   = None
        self.K_vector_ballon  = None
        self.ecs_j_vector     = None
        self.journee_terminee = None

        self.listen_event(self._on_policy_ready, "policy_ready")
        self._try_load_policy(cfg)

        # ── Planification calee sur la grille reelle (00:00, 00:D_segments, ...) ──
        interval_s = int(cfg.D_segments * 3600)

        now = self.datetime()
        seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
        seconds_to_next = interval_s - (seconds_since_midnight % interval_s)
        if seconds_to_next < 10:          # evite un demarrage quasi-immediat
            seconds_to_next += interval_s
        start = now + datetime.timedelta(seconds=seconds_to_next)

        self.run_every(self._apply_policy, start, interval_s)

        self.log(
            f"PolicyReader initialise (intervalle={interval_s}s, "
            f"prochain segment dans {seconds_to_next}s a {start.strftime('%H:%M:%S')})."
        )

    # ------------------------------------------------------------------

    def _on_policy_ready(self, event_name, data, kwargs):
        self.log("PolicyReader: nouvelle politique disponible, rechargement.")
        cfg = BellmanConfig.from_ha(self)
        self._try_load_policy(cfg)
        # Application immédiate sans attendre le prochain tick
        self.run_in(self._apply_policy, 2)

    def _try_load_policy(self, cfg: BellmanConfig):
        d = cfg.data_dir
        try:
            self.pol_uB           = np.load(d + "pol_uB_mat.npy")
            self.pol_uP           = np.load(d + "pol_uP_mat.npy")
            self.S_vector         = np.load(d + "S_vector.npy")
            self.D_vector         = np.load(d + "D_vector.npy")
            self.K_vector_pompe   = np.load(d + "K_vector_pompe.npy")
            self.K_vector_ballon  = np.load(d + "K_vector_ballon.npy")
            self.ecs_j_vector     = np.load(d + "ecs_j_vector.npy")
            self.journee_terminee = np.load(d + "journee_terminee_vector.npy")
            self.log(f"PolicyReader: politique chargee (J_max={self.pol_uB.shape[0]-1}).")
        except FileNotFoundError as exc:
            self.log(f"PolicyReader: {exc} - en attente de policy_ready.", level="WARNING")

    # ------------------------------------------------------------------

    def _apply_policy(self, kwargs):
        if self.pol_uB is None:
            self.log("PolicyReader: politique non disponible, segment ignore.", level="WARNING")
            return

        try:
            self._do_apply()
        except Exception as exc:
            self.log(f"PolicyReader: ERREUR dans _apply_policy - {exc}", level="ERROR")

    def _do_apply(self):
        cfg = BellmanConfig.from_ha(self)

        # ── Index j depuis minuit ──────────────────────────────────
        now = self.datetime()
        h   = now.hour * 2 + (1 if now.minute >= 30 else 0)
        seg = (now.minute % 30) // int(30 / cfg.N_segments)
        j   = h * cfg.N_segments + seg + 1

        J_max = self.pol_uB.shape[0] - 1
        if j > J_max:
            self.log(f"PolicyReader: j={j} > J_max={J_max}, ignore.", level="WARNING")
            return

        self.log(f"PolicyReader: application segment j={j} (h={h}, seg={seg})")

        # ── Etat courant depuis HA ─────────────────────────────────────
        S_cur        = self._read_float(cfg.entity_ballon_energie, cfg.E_min)
        D_cur        = self._read_float(cfg.entity_energie_pompe,  0.0)/cfg.P_nom_P
        K_pompe_cur  = int(self._read_float(cfg.entity_compteur_k_pompe, 0.0))
        K_ballon_cur = int(self._read_float(cfg.entity_compteur_k_ballon, 0.0))

        # ── Snap au plus proche etat discret ───────────────────────────
        s_idx        = int(np.argmin(np.abs(self.S_vector - S_cur)))
        d_idx        = int(np.argmin(np.abs(self.D_vector - D_cur)))
        k_idx_pompe  = int(np.argmin(np.abs(self.K_vector_pompe - K_pompe_cur)))
        k_idx_ballon = int(np.argmin(np.abs(self.K_vector_ballon - K_ballon_cur)))

        uB = int(self.pol_uB[j, s_idx, k_idx_ballon])
        uP = int(self.pol_uP[j, d_idx, k_idx_pompe])

        self.log(
            f"  Etat : S={S_cur:.3f}kWh (K_ballon={K_ballon_cur})  "
            f"D={D_cur:.3f}h  K_pompe={K_pompe_cur} "
            f"(idx s={s_idx} k_ballon={k_idx_ballon} d={d_idx} k={k_idx_pompe})"
        )
        self.log(f"  Commande : uB={uB} (chauffe-eau)  uP={uP} (pompe)")

        # ── Commandes ──────────────────────────────────────────────────
        mode_auto_ballon = (
            str(
                self._read_raw_state(
                    cfg.entity_mode_auto_ballon, 
                    "Automatique")
            ).strip().lower()
            == "automatique"
        )
        if mode_auto_ballon:
            self._switch(cfg.entity_switch_ballon, uB, "Chauffe-eau")
        else:
            self.log("  Mode automatique du chauffe-eau desactive")

        mode_auto_piscine = (
            str(
                self._read_raw_state(
                    cfg.entity_mode_auto_piscine, 
                    "Automatique")
            ).strip().lower() 
            == "automatique"
        )
        if mode_auto_piscine:
            self._switch(cfg.entity_switch_pompe,  uP, "Pompe piscine")
        else:
            self.log("  Mode automatique de la piscine desactive")

        # ── Transitions d'etat ─────────────────────────────────────────
        ecs_j = float(self.ecs_j_vector[j]) if j < len(self.ecs_j_vector) else 0.0
        uB_reel = 1 if self.get_state(cfg.entity_switch_ballon) == "on" else 0
        uP_reel = 1 if self.get_state(cfg.entity_switch_pompe) == "on" else 0

        S_new = (
            S_cur
            + cfg.P_nom_B * uB_reel * cfg.D_segments
            - ecs_j * cfg.n_personnes * cfg.D_segments / 0.5
            - cfg.E_pertes_min_dt
            + cfg.alpha_pertes * cfg.E_min
        ) / (1.0 + cfg.alpha_pertes)

        fin_jour = bool(self.journee_terminee[j]) if j < len(self.journee_terminee) else False
        D_new        = 0.0 if fin_jour else D_cur + cfg.D_segments * uP_reel
        K_pompe_new  = self._transition_K(K_pompe_cur, uP_reel, cfg.K_max_pompe)
        K_ballon_new = self._transition_K(K_ballon_cur, uB_reel, cfg.K_max_ballon)

        self.log(
            f"  Transition : S {S_cur:.3f}->{S_new:.3f}kWh (K_ballon {K_ballon_cur}->{K_ballon_new}) | "
            f"D {D_cur:.3f}->{D_new:.3f}h | K_pompe {K_pompe_cur}->{K_pompe_new}"
        )

        # ── Mise a jour HA ─────────────────────────────────────────────
        self.call_service("input_number/set_value", entity_id=cfg.entity_ballon_energie, value=round(S_new, 4))
        self.set_state(cfg.entity_compteur_k_pompe, state=int(K_pompe_new), 
                        attributes={"friendly_name": "Compteur temporisation pompe piscine K"})
        self.set_state(cfg.entity_compteur_k_ballon, state=int(K_ballon_new),
                        attributes={"friendly_name": "Compteur temporisation ballon K"})

    # ------------------------------------------------------------------

    def _transition_K(self, K, uP, K_max_pompe):
        if uP == 1:
            return min(K + 1, K_max_pompe) if K > 0 else 1
        else:
            return max(K - 1, -K_max_pompe) if K < 0 else -1

    def _switch(self, entity_id, value, label):
        if value == 1:
            self.turn_on(entity_id)
            self.log(f"  [ON]  {label}")
        else:
            self.turn_off(entity_id)
            self.log(f"  [OFF] {label}")

    def _read_raw_state(self, entity_id, default):
        raw = self.get_state(entity_id)
        if raw in (None, "unavailable", "unknown"):
            self.log(f"EnergyConfig: {entity_id} indisponible -> defaut={default}", level="WARNING")
            return default
        return raw

    def _read_float(self, entity_id, default):
        return float(self._read_raw_state(entity_id, default))