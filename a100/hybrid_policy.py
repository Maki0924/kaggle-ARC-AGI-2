"""Label-free output classification and publication gate, shared with hybrid6."""


def output_stratum(confidence):
    n, _, v2 = confidence
    return "Z" if n == 0 else "O" if n == 1 else "W" if v2 <= 2 else "strong"


def output_priority(confidence):
    """Weak second candidate first; preserve the legacy vote order within a stratum."""
    _, v1, v2 = confidence
    return ({"W": 0, "O": 1, "Z": 2, "strong": 3}[output_stratum(confidence)], v2, v1)


def natural_answer(sample):
    return (sample.get("finish") == "stop" and not sample.get("forced_final")
            and not sample.get("recovery_attempted"))
