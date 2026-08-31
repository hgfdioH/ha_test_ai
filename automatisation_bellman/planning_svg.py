"""
AppDaemon App - planning_svg.py
"""

import hassapi as hass
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from automatisation_bellman.bellman_config import BellmanConfig

SVG_PATH = "/homeassistant/www/planning_optimisation.svg"

COLOR_BALLON_ON   = "#f59e0b"
COLOR_BALLON_OFF  = "#1f2937"
COLOR_POMPE_ON    = "#3b82f6"
COLOR_POMPE_OFF   = "#1f2937"
COLOR_ACHAT       = "#f87171"
COLOR_ZERO        = "#6b7280"
COLOR_SOLAIRE     = "#fde68a"
COLOR_CURSEUR     = "#ef4444"
COLOR_BG          = "#111827"
COLOR_BG_CARD     = "#1f2937"
COLOR_TEXTE       = "#f9fafb"
COLOR_TEXTE_MUTED = "#9ca3af"
COLOR_GRID        = "#374151"

MOIS = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}


class PlanningSVG(hass.Hass):

    def initialize(self):
        self.listen_event(self._on_policy_ready, "policy_ready")

        cfg = BellmanConfig.from_ha(self)
        self.D_segments = cfg.D_segments
        interval_s = max(int(self.D_segments * 3600), 5 * 60)  # Minimum 5 minutes
        now  = self.datetime()
        secs = now.hour * 3600 + now.minute * 60 + now.second
        delay = interval_s - (secs % interval_s)
        start = now + timedelta(seconds=delay)
        self.run_every(self._generate, start, interval_s)
        self.run_in(self._generate, 5)
        self.log("PlanningSVG initialise.")

    def _on_policy_ready(self, event_name, data, kwargs):
        self.run_in(self._generate, 3)

    def _generate(self, kwargs):
        try:
            self._do_generate()
        except Exception as exc:
            self.log(f"PlanningSVG: ERREUR - {exc}", level="ERROR")

    # ------------------------------------------------------------------

    def _do_generate(self):
        cfg = BellmanConfig.from_ha(self)
        d   = cfg.data_dir

        pol_uB           = np.load(d + "pol_uB_mat.npy")
        pol_uP           = np.load(d + "pol_uP_mat.npy")
        S_vector         = np.load(d + "S_vector.npy")
        D_vector         = np.load(d + "D_vector.npy")
        K_vector_pompe   = np.load(d + "K_vector_pompe.npy")
        K_vector_ballon  = np.load(d + "K_vector_ballon.npy")
        surplus_j_vector = np.load(d + "surplus_j_vector.npy")
        ecs_j_vector     = np.load(d + "ecs_j_vector.npy")

        S_cur        = float(self._get_initial_state(cfg.entity_ballon_energie, 0))
        D_cur        = 0
        K_pompe_cur  = float(self._get_initial_state(cfg.entity_compteur_k_pompe, 0))
        K_ballon_cur = float(self._get_initial_state(cfg.entity_compteur_k_ballon, 0))

        J_max  = pol_uB.shape[0] - 1
        uB_day = []
        uP_day = []
        solar_day = []
        grid_day  = []

        # ── Heure locale via AppDaemon (respecte le fuseau du serveur HA) ──
        now = self.datetime()   # datetime local du serveur HA
        # +1 : j_now doit etre exprime dans la meme convention 1-based que
        # la boucle ci-dessous et que les matrices pol_uB/pol_uP (j=1 est
        # le premier segment de la journee, cf. policy_reader.py).
        j_now = now.hour * 2 * cfg.N_segments + int(now.minute / (self.D_segments * 60)) + 1
        heure_str = now.strftime("%H:%M")
        date_str = f"{now.day} {MOIS[now.month]} {now.year}"

        # ── Historique des interrupteurs, une SEULE requete par entite ────
        # (au lieu d'une requete par segment -> c'etait la cause du warning
        # AppDaemon "Excessive time spent in callback" : jusqu'a une
        # centaine de requetes d'historique de 24h a chaque generation).
        now_paris = datetime.now(ZoneInfo("Europe/Paris"))
        day_start = now_paris.replace(hour=0, minute=0, second=0, microsecond=0)
        hist_ballon = self._fetch_history_range(
            cfg.entity_switch_ballon, day_start - timedelta(hours=24), now_paris
        )
        hist_pompe = self._fetch_history_range(
            cfg.entity_switch_pompe, day_start - timedelta(hours=24), now_paris
        )

        # ── Modes manuels ("hors automatique") ────────────────────────────
        # En mode automatique, uB/uP viennent de l'historique reel (passe)
        # ou de la politique calculee (futur), comme avant. En mode force,
        # la politique n'est plus pilotee par l'algorithme : on l'affiche
        # differemment (bandeau) plutot que de laisser croire que
        # l'optimisation a pris cette decision.
        mode_ballon = self._classify_mode(cfg.entity_mode_auto_ballon,
                                           self._read_raw_state(cfg.entity_mode_auto_ballon, "Automatique"))
        mode_pompe = self._classify_mode(cfg.entity_mode_auto_piscine,
                                          self._read_raw_state(cfg.entity_mode_auto_piscine, "Automatique"))

        for j in range(1, 48 * cfg.N_segments + 1):
            if j > J_max:
                uB_day.append(0)
                uP_day.append(0)
                solar_day.append(0.0)
                grid_day.append(0.0)
                continue

            h = ((j - 1) // cfg.N_segments) % 48
            i = (j - 1) % cfg.N_segments
            hours = h // 2
            minutes = int(h % 2 * 30 + 30 / cfg.N_segments * i) % 60
            target = datetime.now(ZoneInfo("Europe/Paris")).replace(
                hour=hours, minute=minutes, second=0, microsecond=0
            )
            # Milieu du segment plutot que son debut exact : un changement
            # d'etat HA a 07:15:00.4xx est bien "dans" le segment 07:15,
            # mais echoue un test "<= 07:15:00.000000" pile a la seconde.
            target_mid = target + timedelta(hours=self.D_segments / 2)

            # ── Ballon ──────────────────────────────────────────────────
            if mode_ballon == "off":
                uB = 0
            elif mode_ballon == "on":
                # Marche forcee, thermostat interne : on ne sait pas quand
                # il chauffe reellement -> exclu du calcul (voir _build_svg,
                # bandeau au lieu du planning normal).
                uB = 0
            elif j < j_now:
                uB = self._state_at(hist_ballon, target_mid, "off") == "on"
            else:
                s_idx        = int(np.argmin(np.abs(S_vector - S_cur)))
                k_idx_ballon = int(np.argmin(np.abs(K_vector_ballon - K_ballon_cur)))
                uB = int(pol_uB[j, s_idx, k_idx_ballon])

            # ── Pompe ───────────────────────────────────────────────────
            if mode_pompe == "off":
                uP = 0
            elif mode_pompe == "on":
                # Marche forcee = ON en continu, c'est deterministe (pas de
                # thermostat cache), donc calculable normalement.
                uP = 1
            elif j < j_now:
                uP = self._state_at(hist_pompe, target_mid, "off") == "on"
            else:
                if j == j_now:
                    D_cur = self._read_float(cfg.entity_energie_pompe, 0) / cfg.P_nom_P
                d_idx       = int(np.argmin(np.abs(D_vector - D_cur)))
                k_idx_pompe = int(np.argmin(np.abs(K_vector_pompe - K_pompe_cur)))
                uP = int(pol_uP[j, d_idx, k_idx_pompe])

            uB_day.append(uB)
            uP_day.append(uP)

            E_tot = (uB * cfg.P_nom_B + uP * cfg.P_nom_P) * self.D_segments
            surplus = float(surplus_j_vector[j]) * self.D_segments

            solar_used = min(E_tot, surplus)
            grid_used  = max(E_tot - surplus, 0)

            solar_day.append(solar_used)
            grid_day.append(grid_used)

            # Transition
            ecs_j = float(ecs_j_vector[j])
            S_cur = (
                S_cur
                + cfg.P_nom_B * uB * self.D_segments
                - ecs_j * cfg.n_personnes * self.D_segments / 0.5
                - cfg.E_pertes_min_dt
                + cfg.alpha_pertes * cfg.E_min
            ) / (1.0 + cfg.alpha_pertes)
            D_cur = D_cur + self.D_segments * uP
            K_pompe_cur  = self._transition_K(K_pompe_cur, uP, cfg.K_max_pompe)
            K_ballon_cur = self._transition_K(K_ballon_cur, uB, cfg.K_max_ballon)

        if not any(uB_day) and not any(uP_day) and not any(solar_day) and not any(grid_day):
            self.log(
                "PlanningSVG: ATTENTION - toutes les valeurs du planning sont a zero "
                f"(J_max={J_max}, S_cur_fin={S_cur:.3f}). Verifiez que pol_uB_mat/pol_uP_mat "
                "ont bien ete regeneres avec la resolution/config actuelle (PolicyCreator).",
                level="WARNING",
            )

        svg = self._build_svg(
            uB_day, uP_day, solar_day, grid_day,
            j_now, heure_str, date_str, cfg.N_segments, self.D_segments,
            mode_ballon, mode_pompe,
        )
        with open(SVG_PATH, "w", encoding="utf-8") as f:
            f.write(svg)

        self.log(f"PlanningSVG: genere ({heure_str})")

    # ------------------------------------------------------------------
    # Construction du SVG
    # ------------------------------------------------------------------

    def _build_svg(self, uB, uP, solar, grid, j_now, heure_str, date_str, N_segments, D_segments,
                   mode_ballon="auto", mode_pompe="auto"):
        n = len(uB)

        W, H   = 900, 430
        PAD_L, PAD_R   = 60, 20
        PAD_TOP, PAD_BOT = 56, 66
        CHART_W = W - PAD_L - PAD_R
        CHART_H = H - PAD_TOP - PAD_BOT

        ROW_COURBE   = int(CHART_H * 0.54)
        GAP          = int(CHART_H * 0.06)
        ROW_TIMELINE = int(CHART_H * 0.17)

        y_courbe_top = PAD_TOP
        y_courbe_bot = y_courbe_top + ROW_COURBE
        y_ballon_top = y_courbe_bot + GAP
        y_ballon_bot = y_ballon_top + ROW_TIMELINE
        y_pompe_top  = y_ballon_bot + GAP
        y_pompe_bot  = y_pompe_top + ROW_TIMELINE
        y_chart_bot  = y_pompe_bot
        y_axis       = y_chart_bot + 18
        y_legend     = y_axis + 26

        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
        lines.append(f'<rect width="{W}" height="{H}" rx="12" fill="{COLOR_BG}"/>')

        lines.append(
            f'<text x="{PAD_L}" y="24" fill="{COLOR_TEXTE}" font-family="sans-serif" '
            f'font-size="15" font-weight="bold">Planning d\'optimisation du {date_str}</text>'
        )
        lines.append(
            f'<text x="{W - PAD_R}" y="24" fill="{COLOR_TEXTE_MUTED}" font-family="sans-serif" '
            f'font-size="11" text-anchor="end">Mise a jour : {heure_str}</text>'
        )

        if n == 0:
            lines.append(
                f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="{COLOR_TEXTE_MUTED}" '
                f'font-family="sans-serif" font-size="13" text-anchor="middle">'
                f'Aucune politique disponible pour le moment</text>'
            )
            lines.append("</svg>")
            return "\n".join(lines)

        # Fonds des zones
        for y, h_row in [(y_courbe_top, ROW_COURBE), (y_ballon_top, ROW_TIMELINE), (y_pompe_top, ROW_TIMELINE)]:
            lines.append(f'<rect x="{PAD_L}" y="{y}" width="{CHART_W}" height="{h_row}" rx="4" fill="{COLOR_BG_CARD}"/>')

        bar_w = CHART_W / n

        def x(i):
            return PAD_L + i * bar_w

        def xm(i):
            return x(i) + bar_w / 2

        # Grille verticale (toute la hauteur du graphique)
        for i in range(0, n + 1, max(1, N_segments * 4)):
            xg = x(i)
            lines.append(
                f'<line x1="{xg:.1f}" y1="{y_courbe_top}" x2="{xg:.1f}" y2="{y_chart_bot}" '
                f'stroke="{COLOR_GRID}" stroke-width="0.5" stroke-dasharray="3,3"/>'
            )

        # ── Zone 1 : courbe solaire (au-dessus) / reseau (en-dessous) ────
        energy_max = max(max(solar, default=0), max(grid, default=0), 0.01)
        half = ROW_COURBE // 2
        y_zero = y_courbe_top + half

        def y_up(v):
            return y_zero - (v / energy_max * half)

        def y_down(v):
            return y_zero + (v / energy_max * half)

        pts_solar = [f"{x(0):.1f},{y_zero:.1f}"]
        pts_solar += [f"{xm(i):.1f},{y_up(solar[i]):.1f}" for i in range(n)]
        pts_solar.append(f"{x(n):.1f},{y_zero:.1f}")
        lines.append(f'<path d="M {" L ".join(pts_solar)} Z" fill="{COLOR_SOLAIRE}" opacity="0.35"/>')
        lines.append(
            f'<path d="M {" L ".join(pts_solar[1:-1])}" fill="none" '
            f'stroke="{COLOR_SOLAIRE}" stroke-width="1.5"/>'
        )

        pts_grid = [f"{x(0):.1f},{y_zero:.1f}"]
        pts_grid += [f"{xm(i):.1f},{y_down(grid[i]):.1f}" for i in range(n)]
        pts_grid.append(f"{x(n):.1f},{y_zero:.1f}")
        lines.append(f'<path d="M {" L ".join(pts_grid)} Z" fill="{COLOR_ACHAT}" opacity="0.35"/>')
        lines.append(
            f'<path d="M {" L ".join(pts_grid[1:-1])}" fill="none" '
            f'stroke="{COLOR_ACHAT}" stroke-width="1.5"/>'
        )

        lines.append(
            f'<line x1="{PAD_L}" y1="{y_zero:.1f}" x2="{PAD_L + CHART_W}" y2="{y_zero:.1f}" '
            f'stroke="{COLOR_ZERO}" stroke-width="1"/>'
        )

        # ── Zones 2 et 3 : timelines ballon / pompe (rectangles fusionnes) ─
        # ou bandeau explicatif si le mode n'est pas automatique.
        MSG_BALLON_OFF = "Chauffe-eau en arrêt forcé (mode manuel)"
        MSG_BALLON_ON  = ("Chauffe-eau en marche forcée — géré par le thermostat interne, "
                           "consommation non estimable")
        MSG_POMPE_OFF  = "Pompe en arrêt forcé (mode manuel)"
        MSG_POMPE_ON   = "Pompe en marche forcée (mode manuel)"

        if mode_ballon == "off":
            self._draw_mode_banner(lines, y_ballon_top, ROW_TIMELINE, x, n, MSG_BALLON_OFF, COLOR_TEXTE_MUTED)
        elif mode_ballon == "on":
            self._draw_mode_banner(lines, y_ballon_top, ROW_TIMELINE, x, n, MSG_BALLON_ON, COLOR_BALLON_ON)
        else:
            self._draw_timeline(lines, uB, x, y_ballon_top, ROW_TIMELINE, D_segments,
                                 COLOR_BALLON_ON, COLOR_BALLON_OFF)

        if mode_pompe == "off":
            self._draw_mode_banner(lines, y_pompe_top, ROW_TIMELINE, x, n, MSG_POMPE_OFF, COLOR_TEXTE_MUTED)
        elif mode_pompe == "on":
            self._draw_mode_banner(lines, y_pompe_top, ROW_TIMELINE, x, n, MSG_POMPE_ON, COLOR_POMPE_ON)
        else:
            self._draw_timeline(lines, uP, x, y_pompe_top, ROW_TIMELINE, D_segments,
                                 COLOR_POMPE_ON, COLOR_POMPE_OFF)

        # Labels gauche
        for label, y_mid in [
            ("Énergie", y_zero),
            ("Ballon", y_ballon_top + ROW_TIMELINE / 2),
            ("Pompe",  y_pompe_top + ROW_TIMELINE / 2),
        ]:
            lines.append(
                f'<text x="{PAD_L - 8}" y="{y_mid + 4:.1f}" fill="{COLOR_TEXTE_MUTED}" '
                f'font-family="sans-serif" font-size="10" text-anchor="end">{label}</text>'
            )

        # ── Curseur heure courante ──────────────────────────────────────
        j_now_idx = j_now - 1  # j_now est 1-based, l'axe des segments est 0-based
        if 0 <= j_now_idx < n:
            xc = x(j_now_idx)
            lines.append(
                f'<line x1="{xc:.1f}" y1="{PAD_TOP}" x2="{xc:.1f}" y2="{y_chart_bot}" '
                f'stroke="{COLOR_CURSEUR}" stroke-width="1.5"/>'
            )
            lines.append(
                f'<polygon points="{xc:.1f},{PAD_TOP - 6} {xc-5:.1f},{PAD_TOP-14} {xc+5:.1f},{PAD_TOP-14}" '
                f'fill="{COLOR_CURSEUR}"/>'
            )

        # ── Labels horaires ─────────────────────────────────────────────
        for i in range(0, 48 + 1, 4):
            xg = x(i * N_segments)
            lbl = f"{(i // 2):02d}h"
            lines.append(
                f'<text x="{xg:.1f}" y="{y_axis}" fill="{COLOR_TEXTE_MUTED}" font-family="sans-serif" '
                f'font-size="10" text-anchor="middle">{lbl}</text>'
            )

        # ── Legende ─────────────────────────────────────────────────────
        items = [
            (COLOR_SOLAIRE, "Énergie solaire"),
            (COLOR_ACHAT,   "Énergie réseau"),
            (COLOR_BALLON_ON, "Chauffe-eau ON"),
            (COLOR_POMPE_ON,  "Pompe ON"),
        ]
        leg_x = PAD_L
        for color, label in items:
            lines.append(f'<rect x="{leg_x}" y="{y_legend - 9}" width="10" height="10" fill="{color}" rx="2"/>')
            lines.append(
                f'<text x="{leg_x + 13}" y="{y_legend}" fill="{COLOR_TEXTE_MUTED}" '
                f'font-family="sans-serif" font-size="10">{label}</text>'
            )
            leg_x += 158

        lines.append("</svg>")
        return "\n".join(lines)

    def _draw_mode_banner(self, lines, y_top, row_h, x, n, message, accent_color):
        """Bandeau explicatif a la place de la timeline normale, quand
        l'appareil est en mode manuel (pas piloté par l'optimisation)."""
        x_start, x_end = x(0), x(n)
        lines.append(
            f'<rect x="{x_start:.1f}" y="{y_top:.1f}" width="{x_end - x_start:.1f}" '
            f'height="{row_h:.1f}" rx="6" fill="{COLOR_BG_CARD}" '
            f'stroke="{accent_color}" stroke-width="1.5" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<text x="{(x_start + x_end) / 2:.1f}" y="{y_top + row_h / 2 + 4:.1f}" fill="{accent_color}" '
            f'font-family="sans-serif" font-style="italic" font-size="11" text-anchor="middle">{message}</text>'
        )

    def _draw_timeline(self, lines, values, x, y_top, row_h, D_segments, color_on, color_off):
        """Dessine une piste horizontale avec UN rectangle par periode ON
        continue (fusion des segments consecutifs), plutot qu'un rectangle
        par segment."""
        n = len(values)
        x_start = x(0)
        x_end   = x(n)
        pad_v   = row_h * 0.18
        y0, y1  = y_top + pad_v, y_top + row_h - pad_v

        # Piste de fond
        lines.append(
            f'<rect x="{x_start:.1f}" y="{y0:.1f}" width="{x_end - x_start:.1f}" '
            f'height="{y1 - y0:.1f}" rx="{(y1-y0)/2:.1f}" fill="{color_off}" opacity="0.35"/>'
        )

        for start, end in self._runs(values):
            xa, xb = x(start), x(end)
            w = xb - xa
            lines.append(
                f'<rect x="{xa:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{y1 - y0:.1f}" '
                f'rx="{(y1-y0)/2:.1f}" fill="{color_on}"/>'
            )
            if w >= 70:
                label = f"{self._fmt_hm(start * D_segments)} – {self._fmt_hm(end * D_segments)}"
                lines.append(
                    f'<text x="{(xa+xb)/2:.1f}" y="{(y0+y1)/2 + 4:.1f}" fill="{COLOR_BG}" '
                    f'font-family="sans-serif" font-size="10" font-weight="bold" '
                    f'text-anchor="middle">{label}</text>'
                )

    @staticmethod
    def _runs(values):
        """Renvoie la liste des (start, end) [end exclusif] des segments
        consecutifs a 1 (vrai) dans `values`."""
        runs = []
        start = None
        for i, v in enumerate(values):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(values)))
        return runs

    @staticmethod
    def _fmt_hm(hours_from_midnight):
        total_min = round(hours_from_midnight * 60)
        return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"

    # ------------------------------------------------------------------

    def _classify_mode(self, entity_id, raw_state):
        """Normalise un input_select de mode en 'auto' / 'off' / 'on'.

        ATTENTION : les libelles exacts ci-dessous sont une estimation
        raisonnable (Automatique / Eteint / On) -- s'ils ne correspondent
        pas aux options reelles de vos input_select, le mode restera
        classe 'auto' par defaut et un warning apparaitra dans les logs
        avec la valeur brute recue, pour ajustement facile.
        """
        m = str(raw_state).strip().lower()
        if m == "automatique":
            return "auto"
        if m in ("arrêté", "arrêtée"):
            return "off"
        if m == "marche forcée":
            return "on"
        self.log(
            f"PlanningSVG: mode inconnu pour {entity_id} = '{raw_state}' - traite comme "
            "'automatique'. Si ce n'est pas le comportement attendu, indiquez le libelle "
            "exact de vos options input_select.",
            level="WARNING",
        )
        return "auto"

    def _transition_K(self, K, u, K_max):
        if u == 1:
            return min(K + 1, K_max) if K > 0 else 1
        else:
            return max(K - 1, -K_max) if K < 0 else -1

    def _fetch_history_range(self, entity_id, start_time, end_time):
        """Recupere l'historique d'une entite en UNE SEULE requete pour toute
        la plage utile, a reutiliser en memoire pour chaque segment (voir
        _state_at) plutot que de refaire une requete par segment."""
        history = self.get_history(entity_id=entity_id, start_time=start_time, end_time=end_time)
        if not history or not history[0]:
            self.log(f"Aucun historique pour {entity_id} sur la plage demandee", level="WARNING")
            return []
        return history[0]

    @staticmethod
    def _state_at(history_states, target, default):
        """Retourne l'etat actif A l'instant `target`.

        Un historique HA est une fonction en ESCALIER : une entree reste
        valide jusqu'au changement suivant. Il faut donc prendre le DERNIER
        changement survenu a ou avant `target`, pas celui dont le timestamp
        est le plus proche en valeur absolue -- sinon un changement futur
        proche de `target` peut "remonter" et faire croire qu'il a eu lieu
        plus tot que la realite (ex: un ON a 4h se retrouve applique des
        2h, au milieu de la plage OFF qui precede).
        """
        if not history_states:
            return default
        prior = [s for s in history_states if s["last_changed"] <= target]
        if prior:
            return max(prior, key=lambda s: s["last_changed"])['state']
        # Rien avant target dans la fenetre recuperee (ne devrait arriver
        # que si la fenetre de recherche est trop courte) : on prend la
        # plus ancienne valeur connue plutot que de se rabattre sur `default`.
        return min(history_states, key=lambda s: s["last_changed"])['state']

    def _get_state_at_date(self, entity_id, target, default):
        history_states = self._fetch_history_range(
            entity_id, target - timedelta(hours=24), target + timedelta(hours=self.D_segments)
        )
        return self._state_at(history_states, target, default)

    def _get_initial_state(self, entity_id, default):
        target = datetime.now(ZoneInfo("Europe/Paris")).replace(hour=0, minute=0, second=0, microsecond=0)
        return self._get_state_at_date(entity_id, target, default)

    def _read_raw_state(self, entity_id, default):
        raw = self.get_state(entity_id)
        if raw in (None, "unavailable", "unknown"):
            self.log(f"PlanningSVG: {entity_id} indisponible -> defaut={default}", level="WARNING")
            return default
        return raw

    def _read_float(self, entity_id, default):
        raw = self.get_state(entity_id)
        if raw in (None, "unavailable", "unknown"):
            self.log(f"PlanningSVG: {entity_id} indisponible -> defaut={default}", level="WARNING")
            return default
        return float(raw)