"""Settings shared by the studies here, so they cannot drift apart.

Solver tolerance. Adjustable: raise for validation work, lower for speed. At 1e-6 the
worst observable (internal pressure) carries 2.8e-05 relative error and the radius
3.4e-07, against experimental noise of ~2e-02 -- and the model selection is unchanged
against 1e-9.
"""

RTOL, ATOL = 1e-6, 1e-8
