"""Single-product inventory control as a finite MDP.

Each period: observe stock x (0..capacity), order q units (arrives at once;
orders are capped so stock never exceeds capacity), demand D arrives, you sell
min(x + q, D), unmet demand is lost. Reward = price * sales - order cost
(fixed cost if q > 0, plus unit cost * q) - holding cost * leftover stock
- lost-sales penalty * unmet demand. Demand is Poisson(demand_mean) truncated
at max_demand, with the tail mass folded into max_demand.

The canonical parameters are the defaults; Module 1 participants build these
arrays themselves and compare against prl's fixture.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .tabular import TabularMDP


def demand_pmf(mean: float, max_demand: int) -> np.ndarray:
    """Poisson pmf on 0..max_demand with the upper tail folded into max_demand."""
    k = np.arange(max_demand + 1)
    pmf = poisson.pmf(k, mean)
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


class InventoryMDP(TabularMDP):
    def __init__(
        self,
        capacity: int = 20,
        *,
        demand_mean: float = 6.0,
        max_demand: int = 20,
        price: float = 4.0,
        unit_cost: float = 2.0,
        fixed_cost: float = 5.0,
        holding_cost: float = 0.2,
        lost_sales_penalty: float = 1.0,
        gamma: float = 0.95,
        start: int = 0,
    ):
        self.capacity = capacity
        self.params = dict(
            capacity=capacity,
            demand_mean=demand_mean,
            max_demand=max_demand,
            price=price,
            unit_cost=unit_cost,
            fixed_cost=fixed_cost,
            holding_cost=holding_cost,
            lost_sales_penalty=lost_sales_penalty,
            gamma=gamma,
        )
        pmf = demand_pmf(demand_mean, max_demand)
        self.pmf = pmf
        S = A = capacity + 1
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        for x in range(S):
            for q in range(A):
                q_eff = min(q, capacity - x)
                y = x + q_eff
                order = (fixed_cost if q_eff > 0 else 0.0) + unit_cost * q_eff
                for d, p in enumerate(pmf):
                    sales = min(y, d)
                    left = y - sales
                    r = (
                        price * sales
                        - order
                        - holding_cost * left
                        - lost_sales_penalty * (d - sales)
                    )
                    P[x, q, left] += p
                    R[x, q] += p * r
        super().__init__(
            P=P,
            R=R,
            gamma=gamma,
            start=start,
            state_names=[f"stock {x}" for x in range(S)],
            action_names=[f"order {q}" for q in range(A)],
            meta=dict(self.params),
        )

    def sample_step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        """Sample demand, then return (next stock, realized reward, False)."""
        p = self.params
        q_eff = min(int(a), self.capacity - int(s))
        y = int(s) + q_eff
        d = int(rng.choice(len(self.pmf), p=self.pmf))
        sales = min(y, d)
        left = y - sales
        order = (p["fixed_cost"] if q_eff > 0 else 0.0) + p["unit_cost"] * q_eff
        r = (
            p["price"] * sales
            - order
            - p["holding_cost"] * left
            - p["lost_sales_penalty"] * (d - sales)
        )
        return left, float(r), False
