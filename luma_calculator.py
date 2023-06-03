from typing import Tuple
import numpy as np
from statistics import geometric_mean
from telemetry import LumaTelemetry


class LumaCalculator:
    BLACK_THR, WHITE_THR: float = 9.0, 95.0
    BLACK_LIM, WHITE_LIM: float = 0.6, 0.6

    def calculate(pic: np.ndarray) -> LumaTelemetry:
        luma_vector = LumaCalculator.__picture_luma(pic)
        lightness_vector = [LumaCalculator.__luma_to_perc_lightness(
            pix) for pix in luma_vector]
        filtered_lightness_vector = LumaCalculator.__filter_lightness_vector(
            lightness_vector)

        return LumaTelemetry(*(np.mean(luma_vector), geometric_mean(luma_vector)) + LumaCalculator.__calculate_lightness_values(lightness_vector) + LumaCalculator.__calculate_lightness_values(filtered_lightness_vector))

    @staticmethod
    def __calculate_lightness_values(lightness_vector: np.ndarray) -> Tuple[float, float, float]:
        mean_perc_lightness: float = np.mean(lightness_vector)
        median_perc_lightness: float = np.median(lightness_vector)
        geom_mean_perc_lightness: float = geometric_mean(lightness_vector)
        return (mean_perc_lightness, median_perc_lightness, geom_mean_perc_lightness)

    @staticmethod
    def __linearize_pixel(pixel: np.ndarray) -> np.ndarray:
        def transform(channel: float) -> float:
            if channel < 0.04045:
                return channel / 12.92
            return ((channel + 0.055) / 1.055) ** 2.4

        return np.array([transform(c) for c in pixel])

    @staticmethod
    def __pixel_luma(pixel: np.ndarray) -> float:
        norm = np.array([1/255, 1/255, 1/255])
        lin_pixel = LumaCalculator.__linearize_pixel(pixel * norm)
        multipliers = np.array([0.2126, 0.7152, 0.0722])
        return np.dot(lin_pixel, multipliers)

    @staticmethod
    def __luma_to_perc_lightness(y: float) -> float:
        return y * 903.3 if y <= 0.008856 else y ** (1 / 3) * 116 - 16

    @staticmethod
    def __picture_luma(pic: np.ndarray) -> np.ndarray:
        return np.array([[LumaCalculator.__pixel_luma(x) for x in r] for r in pic]).flatten()

    @staticmethod
    def __bw_lightness_parts(lightness_vector: np.ndarray) -> Tuple[float, float]:
        l_vec_size = len(lightness_vector)
        black = len(
            list(filter(lambda x: x < LumaCalculator.BLACK_THR, lightness_vector))) / l_vec_size
        white = len(
            list(filter(lambda x: x > LumaCalculator.WHITE_THR, lightness_vector))) / l_vec_size
        return (black, white)

    @staticmethod
    def __filter_lightness_vector(lightness_vector: np.ndarray) -> np.ndarray:
        b, w = LumaCalculator.__bw_lightness_parts(lightness_vector)
        filtered = lightness_vector.copy()
        if b < LumaCalculator.BLACK_LIM:
            filtered = np.array(
                list(filter(lambda x: LumaCalculator.BLACK_THR < x, filtered)))
        if w < LumaCalculator.WHITE_LIM:
            filtered = np.array(
                list(filter(lambda x: x < LumaCalculator.WHITE_THR, filtered)))
        return filtered
