import hassapi as hass
from automatisation_bellman.bellman_config import BellmanConfig

class StockUpdater(hass.Hass):

    def initialize(self):
        cfg = BellmanConfig.from_ha(self)

        self.P_nom_B      = cfg.P_nom_B
        self.E_max        = cfg.E_max
        self.entity_power  = cfg.entity_puissance_ballon
        self.entity_switch = cfg.entity_switch_ballon
        self.entity_stock  = cfg.entity_ballon_energie
        self.cooldown_h    = 2

        self._last_update  = None
        self._check_timer  = None

        self.listen_state(self._on_power_change,  self.entity_power)
        self.listen_state(self._on_switch_change, self.entity_switch)
        self.log("StockUpdater initialise.")

    # ------------------------------------------------------------------

    def _on_power_change(self, entity, attribute, old, new, *args, **kwargs):
        """Condition 1 : puissance qui retombe a ~0 alors que la prise est ON."""
        raw = self.get_state(self.entity_power)
        try:
            power = float(raw)
        except (TypeError, ValueError):
            return

        switch_on = self.get_state(self.entity_switch) == "on"

        if power <= self.P_nom_B / 2 and switch_on:
            self._set_stock_max("coupure thermostat (puissance retombee a 0 prise ON)")

    def _on_switch_change(self, entity, attribute, old, new, *args, **kwargs):
        """Condition 2 : prise vient de s'allumer → verifie la puissance dans 10s."""
        if new != "on":
            return
        # Annule un timer precedent si la prise s'est rallumee rapidement
        if self._check_timer is not None:
            self.cancel_timer(self._check_timer)
        self._check_timer = self.run_in(self._check_power_after_on, 10)

    def _check_power_after_on(self, kwargs):
        self._check_timer = None

        if self.get_state(self.entity_switch) != "on":
            return   # prise deja eteinte entre-temps

        raw = self.get_state(self.entity_power)
        try:
            power = float(raw)
        except (TypeError, ValueError):
            return

        if power <= self.P_nom_B / 2:
            self._set_stock_max(f"demarrage sans montee en puissance ({power:.0f}W apres 10s)")

    # ------------------------------------------------------------------

    def _set_stock_max(self, raison):
        if not self._cooldown_ok():
            self.log(f"StockUpdater: cooldown actif, mise a jour ignoree ({raison})")
            return

        self.log(f"StockUpdater: ballon plein -> stock = {self.E_max} kWh ({raison})")
        self.call_service("input_number/set_value", entity_id=self.entity_stock, value=self.E_max)
        self._last_update = self.datetime()

    def _cooldown_ok(self):
        if self._last_update is None:
            return True
        elapsed_h = (self.datetime() - self._last_update).total_seconds() / 3600
        return elapsed_h >= self.cooldown_h