"""Explicit batch/shape limits; bounds are candidates pending GPU validation."""
PROFILES = [
    {"min": [1, 1, 128, 128], "opt": [8, 1, 128, 128], "max": [16, 1, 128, 128]},
    {"min": [1, 1, 256, 256], "opt": [4, 1, 256, 256], "max": [8, 1, 256, 256]},
    {"min": [1, 1, 32, 32], "opt": [1, 1, 128, 128], "max": [1, 1, 512, 512]},
]


def choose_profile(shape, profiles):
    for index, item in enumerate(profiles):
        if len(shape) == len(item['min']) and all(lo <= n <= hi for n, lo, hi in zip(shape, item['min'], item['max'])):
            return index
    raise ValueError(f'No TensorRT profile supports {shape}')
