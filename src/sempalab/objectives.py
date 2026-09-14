from sempalab.metrics import exact_ats, wts


def evaluate_partition(
    partition,
    message,
    encoder,
    objective,
    p,
):
    if objective == "ats":
        return exact_ats(
            partition,
            message,
            encoder,
            p,
        )

    if objective == "wts":
        return wts(
            partition,
            message,
            encoder,
            p,
        )

    raise ValueError(
        f"Unknown objective: {objective}"
    )