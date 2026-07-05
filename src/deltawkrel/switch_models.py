"""Switch-model placeholders and safe validation targets.

The ideal quantum-switch process matrix is deliberately not faked here. The
submission package must implement this object from a published convention and
validate the SDP against a known benchmark before formal submission.
"""
from __future__ import annotations

from .projectors import ProcessDims, white_noise_process


def ideal_quantum_switch(*args, **kwargs):
    """Placeholder for the ideal quantum switch.

    This raises by design to prevent accidental claims of completed benchmark
    validation. Implement from a cited convention before formal submission.
    """
    raise NotImplementedError(
        "ideal_quantum_switch is not implemented yet. Complete this from the cited "
        "process-matrix convention and validate against a published benchmark."
    )


def white_noise_validation_process(dims: ProcessDims = ProcessDims()):
    """A safe CS validation target with Tr(W)=d_O."""
    return white_noise_process(dims)
