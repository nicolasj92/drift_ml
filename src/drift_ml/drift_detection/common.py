class DriftDetector:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
