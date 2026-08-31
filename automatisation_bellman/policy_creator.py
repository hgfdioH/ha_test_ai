"""
AppDaemon App - policy_creator.py
===================================
Balayage rétrograde de Bellman, DECOUPLE en 2 sous-problemes independants :

  - Sous-DP "ballon"  : etat (S, K_ballon)   (N_stock x N_K_ballon)
  - Sous-DP "pompe"   : etat (D, K_pompe)    (N_duree x N_K_pompe)

Chaque sous-probleme porte sa propre contrainte de temporisation
(marche/arret minimum) via un compteur K, exactement le meme mecanisme
pour les deux : K>0 compte le temps ecoule depuis l'allumage, K<0 le
temps ecoule depuis l'extinction, et il est interdit de changer d'etat
avant que |K| atteigne K_max_pompe (cf. _feasible/_transition_K).

Les deux sous-problemes ne sont couples que par le partage du surplus
solaire instantane. On resout cette dependance avec une regle simple et
deterministe (voir _cout_tables) plutot qu'un DP joint sur le produit des
4 dimensions d'etat (S x K_ballon x D x K_pompe) : le cout de calcul passe
d'un produit a une somme, ce qui retire l'essentiel de l'explosion
combinatoire, au prix d'une solution seulement approximativement optimale
(et non plus exacte).

Declenche par l'evenement 'energy_config_ready'.
"""

import hassapi as hass
import time
import numpy as np
import threading

from automatisation_bellman.bellman_config import BellmanConfig
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class PolicyCreator(hass.Hass):

    def initialize(self):
        self._set_status("En attente")
        self.listen_event(self._on_config_ready, "energy_config_ready")
        self.log("PolicyCreator initialise, en attente de 'energy_config_ready'.")

    # ------------------------------------------------------------------

    def _on_config_ready(self, event_name, data, kwargs):
        self.log("PolicyCreator: evenement recu - calcul en arriere-plan.")
        self._set_status("Calculs en cours")
        threading.Thread(
            target=self._safe_sweep, args=(data,), daemon=True
        ).start()

    def _safe_sweep(self, data):
        try:
            self._backward_sweep(data)
        except Exception as exc:
            self.log(f"PolicyCreator: ERREUR - {exc}", level="ERROR")
            self._set_status("Erreur")

    # ------------------------------------------------------------------

    def _backward_sweep(self, data):
        # ── Config partagée ────────────────────────────────────────────
        cfg = BellmanConfig.from_ha(self)

        J = int(data["J"])

        S_0 = self._get_initial_state(cfg.entity_ballon_energie)

        K_max_pompe         = cfg.K_max_pompe
        K_max_ballon  = cfg.K_max_ballon
        D_segments    = cfg.D_segments
        alpha_pertes  = cfg.alpha_pertes
        N_stock       = cfg.N_stock
        N_K_pompe           = cfg.N_K_pompe
        N_K_ballon    = cfg.N_K_ballon

        # ── Chargement des arrays ──────────────────────────────────────
        d = cfg.data_dir
        S_vector                = np.load(d + "S_vector.npy")
        D_vector                = np.load(d + "D_vector.npy")
        D_piscine_min_grid      = D_vector[-1]
        N_duree                 = len(D_vector)
        K_vector_pompe                = np.load(d + "K_vector_pompe.npy")
        K_vector_ballon         = np.load(d + "K_vector_ballon.npy")
        surplus_j_vector        = np.load(d + "surplus_j_vector.npy")
        ecs_j_vector             = np.load(d + "ecs_j_vector.npy")
        journee_terminee_vector = np.load(d + "journee_terminee_vector.npy")

        self.log(
            f"PolicyCreator: J={J}, N_stock={N_stock}, N_K_ballon={N_K_ballon}, "
            f"N_duree={N_duree}, N_K_pompe={N_K_pompe} (DP decouples ballon/pompe)"
        )

        # ── Cout immediat, decouple ─────────────────────────────────────
        cout_ballon, cout_pompe = self._cout_tables(cfg, J, D_segments, surplus_j_vector)

        # ── Transitions S (ballon) ──────────────────────────────────────
        # ecs_j_vector est deja interpole ET mis a l'echelle par
        # conso_perso_ecs (cf. energy_config.py). Le profil se repete a
        # l'identique chaque jour, donc une table sur une seule journee
        # "fine" (pas_par_jour lignes) suffit.
        pas_par_jour = 48 * cfg.N_segments
        S_next = np.full((pas_par_jour, N_stock, 2), -1, dtype=np.int32)
        for i in range(pas_par_jour):
            ecs_seg = ecs_j_vector[i + 1]
            const_ecs = (
                -ecs_seg * cfg.n_personnes * D_segments / 0.5
                - cfg.E_pertes_min_dt
                + alpha_pertes * cfg.E_min
            )
            for ub in (0, 1):
                S_new = 1 / (1 + alpha_pertes) * (
                    S_vector
                    + cfg.P_nom_B * ub * D_segments
                    + const_ecs)

                valid = (S_new >= cfg.E_min) & (S_new <= cfg.E_max)
                idx   = np.clip(
                    np.round((S_new - cfg.E_min) / cfg.step_stock).astype(int),
                    0, N_stock - 1
                )
                S_next[i, :, ub] = np.where(valid, idx, -1)

        # ── Transitions K (ballon) ────────────────────────────────────────
        # Mecanisme identique a celui de la pompe (cf. plus bas), avec ses
        # propres K_max_ballon / D_tempo_ballon.
        K_next_ballon = np.full((N_K_ballon, 2), -(K_max_ballon + 1), dtype=np.int32)
        for k_idx, k in enumerate(K_vector_ballon):
            for ub in (0, 1):
                if self._feasible(int(k), ub, K_max_ballon, D_segments, cfg.D_tempo_ballon):
                    k_new = self._transition_K(int(k), ub, K_max_ballon)
                    K_next_ballon[k_idx, ub] = int(np.argmin(np.abs(K_vector_ballon - k_new)))

        valid_k_ballon        = K_next_ballon >= -K_max_ballon
        switch_k_indices_ballon = np.where((K_vector_ballon == 1) | (K_vector_ballon == -1))[0]

        # ── Transitions D, K (pompe) ─────────────────────────────────────
        D_next = np.empty((2, N_duree, 2), dtype=np.int32)
        for up in (0, 1):
            # En journée
            idx = np.clip(
                np.round((D_vector + up * D_segments) / D_segments).astype(int),
                0, N_duree - 1
            )
            D_next[0, :, up] = idx

            # Debut de journee (remise a zéro)
            idx = np.clip(
                np.round(np.full(N_duree, up * D_segments) / D_segments).astype(int),
                0, N_duree - 1
            )
            D_next[1, :, up] = idx

        K_next = np.full((N_K_pompe, 2), -(K_max_pompe + 1), dtype=np.int32)
        for k_idx, k in enumerate(K_vector_pompe):
            for up in (0, 1):
                if self._feasible(int(k), up, K_max_pompe, D_segments, cfg.D_tempo_pompe):
                    k_new = self._transition_K(int(k), up, K_max_pompe)
                    K_next[k_idx, up] = int(np.argmin(np.abs(K_vector_pompe - k_new)))

        valid_k = K_next >= -K_max_pompe

        # ── Penalites ─────────────────────────────────────────────────
        PENALITE_BOUCLAGE            = 1e6
        PENALITE_COMMUTATION         = cfg.P_nom_P * cfg.D_tempo_pompe * cfg.lambda_reseau / 10
        PENALITE_COMMUTATION_BALLON  = cfg.P_nom_B * cfg.D_tempo_ballon * cfg.lambda_reseau / 10
        switch_k_indices              = np.where((K_vector_pompe == 1) | (K_vector_pompe == -1))[0]

        # Retard sur le crepuscule
        date_crepuscule_utc = datetime.fromisoformat(
            self._read_raw_state(cfg.entity_crepuscule, "2026-06-24T23:59:59+00:00")
        )
        date_crepuscule = date_crepuscule_utc.astimezone(ZoneInfo("Europe/Paris"))

        h_crepuscule   = date_crepuscule.hour * 2 + (1 if date_crepuscule.minute >= 30 else 0)
        seg_crepuscule = (date_crepuscule.minute % 30) // int(30 / cfg.N_segments)
        j_crepuscule   = h_crepuscule * cfg.N_segments + seg_crepuscule + 1

        # ── Etats terminaux, un par sous-probleme ───────────────────────
        # Le ballon vise a revenir pres de S_0 en fin d'horizon, quel que
        # soit son etat de temporisation K_ballon (d'ou le broadcast [:,None]).
        Z_B = np.repeat(
            (PENALITE_BOUCLAGE * (S_vector - S_0) ** 2)[:, None].astype(np.float32),
            N_K_ballon, axis=1
        )
        Z_P = np.zeros((N_duree, N_K_pompe), dtype=np.float32)

        pol_uB = np.zeros((J + 1, N_stock, N_K_ballon), dtype=np.int8)
        pol_uP = np.zeros((J + 1, N_duree, N_K_pompe), dtype=np.int8)

        self.log("PolicyCreator: balayage retrograde demarre (2 DP decouples)...")

        # ── Balayage retrograde de Bellman (les 2 sous-DP dans la meme boucle j) ──
        for j in range(J, 0, -1):
            i_in_day = (j - 1) % pas_par_jour

            # ═══ Sous-DP ballon (etat (S, K_ballon)) ═════════════════════
            Q_B = np.full((2, N_stock, N_K_ballon), np.inf, dtype=np.float32)
            for ub in (0, 1):
                s_nxt       = S_next[i_in_day, :, ub]
                k_nxt       = K_next_ballon[:, ub]
                switch_mask = np.isin(k_nxt, switch_k_indices_ballon)

                future = Z_B[s_nxt[:, None], np.clip(k_nxt, 0, None)[None, :]]
                future = future.copy()
                future[:, switch_mask] += PENALITE_COMMUTATION_BALLON
                future = np.where(s_nxt[:, None] >= 0, future, np.inf)
                future = np.where(valid_k_ballon[:, ub][None, :], future, np.inf)

                Q_B[ub] = future + cout_ballon[j, ub]

            Z_B       = np.min(Q_B, axis=0)
            pol_uB[j] = np.argmin(Q_B, axis=0)

            # ═══ Sous-DP pompe (etat (D, K)) ═════════════════════════════
            if j % pas_par_jour > j_crepuscule:
                Z_P = Z_P + (PENALITE_BOUCLAGE * (D_vector - D_piscine_min_grid) ** 2)[:, None]

            debut = bool(journee_terminee_vector[j - 1]) if j > 1 else False

            Q_P = np.full((2, N_duree, N_K_pompe), np.inf, dtype=np.float32)
            for up in (0, 1):
                d_nxt       = D_next[int(debut), :, up]
                k_nxt       = K_next[:, up]
                switch_mask = np.isin(k_nxt, switch_k_indices)

                future = Z_P[d_nxt[:, None], np.clip(k_nxt, 0, None)[None, :]]
                future = future.copy()
                future[:, switch_mask] += PENALITE_COMMUTATION
                future = np.where(valid_k[:, up][None, :], future, np.inf)

                Q_P[up] = future + cout_pompe[j, up]

            Z_P       = np.min(Q_P, axis=0)
            pol_uP[j] = np.argmin(Q_P, axis=0)

            if j % max(1, J // 10) == 0:
                self.log(f"PolicyCreator: {int(100*(J-j)/J)}% - j={j}/{J}")

            if j % 10 == 0:
                time.sleep(0.001)  # 1ms pour redonner la prio à d'autre threads

        # ── Sauvegarde ─────────────────────────────────────────────────
        np.save(d + "pol_uB_mat", pol_uB)
        np.save(d + "pol_uP_mat", pol_uP)
        self.log(
            f"PolicyCreator: sauvegarde "
            f"(ballon={pol_uB.nbytes/1e6:.2f} Mo, pompe={pol_uP.nbytes/1e6:.2f} Mo)."
        )
        self._set_status("Prêt")
        self.fire_event("policy_ready")

    # ------------------------------------------------------------------

    def _cout_tables(self, cfg, J, D_segments, surplus_j_vector):
        """Cout immediat (EUR) pour chaque sous-probleme, decouple par un
        partage simple et deterministe du surplus solaire instantane :

          - La pompe (faible puissance nominale, contrainte quotidienne
            stricte de duree minimale) a un acces prioritaire complet au
            surplus.
          - Le ballon recoit le surplus restant, avec une reserve fixe de
            P_nom_P deduite par prudence (que la pompe soit reellement en
            train de consommer ou non a cet instant).

        C'est une approximation : la somme cout_ballon + cout_pompe peut
        legerement s'ecarter du cout joint exact (qui dependait de la
        puissance totale simultanee), mais l'ecart reste borne par
        P_nom_P * D_segments * (lambda_reseau - lambda_solaire) par pas de
        temps, ce qui est faible devant P_nom_B.
        """
        surplus_pompe  = surplus_j_vector
        surplus_ballon = np.maximum(surplus_j_vector - cfg.P_nom_P, 0.0)

        cout_ballon = np.zeros((J + 1, 2), dtype=np.float32)
        cout_pompe  = np.zeros((J + 1, 2), dtype=np.float32)

        for ub in (0, 1):
            P     = cfg.P_nom_B * ub
            E_sol = D_segments * np.minimum(surplus_ballon[1:J + 1], P)
            E_res = D_segments * P - E_sol
            cout_ballon[1:J + 1, ub] = E_sol * cfg.lambda_solaire + E_res * cfg.lambda_reseau

        for up in (0, 1):
            P     = cfg.P_nom_P * up
            E_sol = D_segments * np.minimum(surplus_pompe[1:J + 1], P)
            E_res = D_segments * P - E_sol
            cout_pompe[1:J + 1, up] = E_sol * cfg.lambda_solaire + E_res * cfg.lambda_reseau

        return cout_ballon, cout_pompe

    # ------------------------------------------------------------------

    def _transition_K(self, K, u, K_max_pompe):
        if u == 1:
            return min(K + 1, K_max_pompe) if K > 0 else 1
        else:
            return max(K - 1, -K_max_pompe) if K < 0 else -1

    def _feasible(self, K, u, K_max_pompe, D_segments, D_tempo_pompe):
        lim = D_tempo_pompe / D_segments
        if 0 < K < lim and u == 0:
            return False
        if -lim < K < 0 and u == 1:
            return False
        return True

    def _set_status(self, status):
        self.set_state("sensor.energy_opt_policy_status", state=status,
                        attributes={"friendly_name": "Statut politique optimale"})

    def _get_initial_state(self, entity_id):
        target = datetime.now(
            ZoneInfo("Europe/Paris")
        ).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        history_states = self.get_history(
            entity_id=entity_id,
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now()
        )[0]

        closest = min(history_states, key=lambda s: abs(
            s["last_changed"] - target
        ))
        return float(closest['state'])


    def _read_raw_state(self, entity_id, default):
        raw = self.get_state(entity_id)
        if raw in (None, "unavailable", "unknown"):
            self.log(f"EnergyConfig: {entity_id} indisponible -> defaut={default}", level="WARNING")
            return default
        return raw