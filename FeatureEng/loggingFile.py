def run_forward_and_log(*args, **kwargs):
    best_combo, history, best_step = eval_stepwise_forward(*args, **kwargs)

    logger.info(
        f"[FORWARD] Best bands: {best_step['bands']} | "
        f"n_bands={best_step['n_bands']} | "
        f"n_features={best_step['n_features']} | "
        f"score={best_step['score']:.6f}"
    )

    return best_combo, history, best_step

def run_backward_and_log(*args, **kwargs):
    best_combo, history, best_step = eval_stepwise_backward(*args, **kwargs)

    logger.info(
        f"[BACKWARD] Best bands: {best_step['bands']} | "
        f"n_bands={best_step['n_bands']} | "
        f"n_features={best_step['n_features']} | "
        f"score={best_step['score']:.6f}"
    )

    return best_combo, history, best_step

def run_rfe_and_log(*args, **kwargs):
    bandSorted = eval_rfe(*args, **kwargs)

    logger.info(
        f"[RFE] Removed order (worst → best): {bandSorted}"
    )

    return bandSorted

def run_point_biserial_and_log(*args, **kwargs):
    result = eval_point_biserial(*args, **kwargs)

    top5 = list(result.items())[:5]

    logger.info(
        f"[PointBiserial] Top 5 bands: "
        f"{[(k, round(v['corr'], 5)) for k, v in top5]}"
    )

    return result

def run_mi_and_log(*args, **kwargs):
    result = eval_mutual_information(*args, **kwargs)

    top5 = result[-5:]  # vì đang sort tăng dần

    logger.info(
        f"[MutualInfo] Top 5 bands: "
        f"{[(x['feature'], round(x['MI_score'], 6)) for x in top5]}"
    )

    return result
