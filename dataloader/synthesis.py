"""Reflection synthesis models for on-the-fly data augmentation."""

import random
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.stats import truncnorm


class RDNetReflectionSynthesis(object):
    """RDNet-style reflection synthesis with channel-wise truncated-normal scaling."""

    def __init__(self):
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]
        self.sigma_range = [2, 5]
        self.alpha_range = [0.8, 1.0]
        self.beta_range = [0.4, 1.0]

    def __call__(self, T_, R_):
        """Synthesize blended image from PIL transmission and reflection."""
        T_ = np.asarray(T_, np.float32) / 255.
        R_ = np.asarray(R_, np.float32) / 255.

        # Gaussian-blur reflection (defocus)
        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T)
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')
        R_ = np.clip(R_, 0, 1)

        # Channel-wise truncated-normal scaling for transmission
        a1 = truncnorm((0.82 - 1.109) / 0.118, (1.42 - 1.109) / 0.118, loc=1.109, scale=0.118)
        a2 = truncnorm((0.85 - 1.106) / 0.115, (1.35 - 1.106) / 0.115, loc=1.106, scale=0.115)
        a3 = truncnorm((0.85 - 1.078) / 0.116, (1.31 - 1.078) / 0.116, loc=1.078, scale=0.116)

        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        T_[..., 0] *= a1.rvs()
        T_[..., 1] *= a2.rvs()
        T_[..., 2] *= a3.rvs()
        T_ = np.clip(T_, 0, 1)

        T, R = T_, b * R_

        # Screen blend (70 %) or additive with re-normalization (30 %)
        if random.random() < 0.7:
            I = T + R - T * R
        else:
            I = T + R
            if np.max(I) > 1:
                m = I[I > 1]
                m = (np.mean(m) - 1) * 1.3
                I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)

        return T_, R_, I


class PhysicalReflectionSynthesis:
    """Physically-motivated reflection synthesis in linear light space.

    Pipeline:
        sRGB -> linear -> color jitter on T and R -> blur R -> optional ghosting
        -> spatial alpha mask (low-freq noise x vignetting) -> additive blend
        -> exposure normalize -> linear -> sRGB.
    """

    def __init__(self):
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]
        self.sigma_range = [2, 5]

        self.tau_range = [0.15, 0.45]

        self.ghost_prob = 0.5
        self.ghost_weight_range = [0.05, 0.15]
        self.ghost_shift_range = [-15, 15]
        self.ghost_blur_sigma = 2.0

        self.alpha_base_range = [0.3, 0.8]
        self.color_jitter_range = [0.85, 1.15]

    @staticmethod
    def srgb_to_linear(x):
        return np.where(x <= 0.04045,
                        x / 12.92,
                        ((x + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def linear_to_srgb(x):
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.0031308,
                        12.92 * x,
                        1.055 * np.power(x, 1.0 / 2.4) - 0.055)

    def _color_jitter(self, img):
        for c in range(3):
            img[..., c] *= np.random.uniform(*self.color_jitter_range)
        return img

    def _blur_reflection(self, R):
        ks = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(*self.sigma_range)
        k1d = cv2.getGaussianKernel(ks, sigma)
        k2d = np.dot(k1d, k1d.T)
        for c in range(3):
            R[..., c] = convolve2d(R[..., c], k2d, mode='same')
        return np.clip(R, 0, None)

    def _add_ghosting(self, R):
        h, w = R.shape[:2]
        dx = np.random.randint(self.ghost_shift_range[0], self.ghost_shift_range[1] + 1)
        dy = np.random.randint(self.ghost_shift_range[0], self.ghost_shift_range[1] + 1)
        wt = np.random.uniform(*self.ghost_weight_range)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ghost = cv2.warpAffine(R, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        ghost = cv2.GaussianBlur(ghost, (5, 5), self.ghost_blur_sigma)
        return R + wt * ghost

    def _spatial_alpha(self, h, w):
        noise = np.random.uniform(0.6, 1.0, (8, 8)).astype(np.float32)
        alpha = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.GaussianBlur(alpha, (31, 31), 10)

        cy = np.random.uniform(0.3, 0.7)
        cx = np.random.uniform(0.3, 0.7)
        Y, X = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        vig_str = np.random.uniform(0.0, 0.4)
        vignette = 1.0 - vig_str * (dist / (dist.max() + 1e-8))

        alpha = alpha * vignette
        base = np.random.uniform(*self.alpha_base_range)
        alpha = base * (alpha / (alpha.mean() + 1e-8))
        alpha = np.clip(alpha, 0.05, 1.0)

        return alpha[..., np.newaxis].astype(np.float32)

    def __call__(self, T_pil, R_pil):
        """Args: PIL Images (uint8, sRGB). Returns (T, R, I) numpy float32 [0,1] sRGB."""
        h, w = np.asarray(T_pil).shape[:2]

        T = self.srgb_to_linear(np.asarray(T_pil, np.float32) / 255.0).copy()
        R = self.srgb_to_linear(np.asarray(R_pil, np.float32) / 255.0).copy()

        T = self._color_jitter(T)
        R = self._color_jitter(R)

        R = self._blur_reflection(R)
        if random.random() < self.ghost_prob:
            R = self._add_ghosting(R)

        alpha = self._spatial_alpha(h, w)

        T_norm = T / (T.mean() + 1e-8)
        R_norm = R / (R.mean() + 1e-8)
        I_raw = T_norm + alpha * R_norm

        tau = np.random.uniform(*self.tau_range)
        e = tau / (I_raw.mean() + 1e-8)

        T_lin = e * T_norm
        R_lin = e * alpha * R_norm
        I_lin = T_lin + R_lin

        T_out = self.linear_to_srgb(np.clip(T_lin, 0, 1)).astype(np.float32)
        R_out = self.linear_to_srgb(np.clip(R_lin, 0, 1)).astype(np.float32)
        I_out = self.linear_to_srgb(np.clip(I_lin, 0, 1)).astype(np.float32)

        return T_out, R_out, I_out
