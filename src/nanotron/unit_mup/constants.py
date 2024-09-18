HIDDEN_WEIGHT_SCALE_POWER = (0.5, 0.5, 0.5)
LM_HEAD_SCALE_POWER = (1.0, 0.5, 0.5)


WEIGHT_TYPE_TO_SCALE_POWER = {
    "hidden": HIDDEN_WEIGHT_SCALE_POWER,
    "output": LM_HEAD_SCALE_POWER,
}

WEIGHT_TYPE_TO_CONSTRAINT = {
    "hidden": "to_output_scale",
    # "output": None
    "output": None,
}
