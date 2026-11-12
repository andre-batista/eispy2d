
import pickle

import numpy as np


class BayesNetwork:

    N_IF_BINS = 80
    N_SF_BINS = 80
    N_CHI_BINS = 80

    IF_RANGE = (0.0, 1.0)
    SF_RANGE = (0.0, 1.0)
    CHI_RANGE = (0.0, 0.25)

    SF_NORMALIZE = "max"        
    ALPHA = 1.0
    MIN_COUNT = 5
    DECISION = "mean"

    def __init__(self, n_if_bins=None, n_sf_bins=None, n_chi_bins=None,
                 if_range=None, sf_range=None, chi_range=None,
                 alpha=None, min_count=None, decision=None,
                 sf_normalize="keep"):
        if n_if_bins is not None:
            self.N_IF_BINS = n_if_bins
        if n_sf_bins is not None:
            self.N_SF_BINS = n_sf_bins
        if n_chi_bins is not None:
            self.N_CHI_BINS = n_chi_bins
        if if_range is not None:
            self.IF_RANGE = if_range
        if sf_range is not None:
            self.SF_RANGE = sf_range
        if chi_range is not None:
            self.CHI_RANGE = chi_range
        if alpha is not None:
            self.ALPHA = alpha
        if min_count is not None:
            self.MIN_COUNT = min_count
        if decision is not None:
            self.DECISION = decision
        if sf_normalize != "keep":
            self.SF_NORMALIZE = sf_normalize

        # IF=i, SF=s, CHI=k
        self.counts = np.zeros(
            (self.N_IF_BINS, self.N_SF_BINS, self.N_CHI_BINS), dtype=np.int64)
        self.n_instances = 0

    def pixel_features(self, scf, idf, gs):
        if gs is None:
            raise ValueError("gs é necessário para o SF por pixel "
                             "(retroprojeção do campo espalhado).")
        scf = np.asarray(scf)
        idf = np.asarray(idf)
        gs = np.asarray(gs)
        G = gs if gs.shape[0] == scf.shape[0] else gs.T

        if_vals = np.abs(idf).mean(axis=1)                    # (N,)
        sf_vals = np.abs(G.conj().T @ scf).mean(axis=1)       # (N,)
        if self.SF_NORMALIZE == "max":
            sf_vals = sf_vals / (sf_vals.max() + 1e-30)
        return if_vals, sf_vals

    @staticmethod
    def _bin(x, rng, n):
        lo, hi = rng
        x = np.nan_to_num(np.asarray(x, dtype=float),
                          nan=lo, posinf=hi, neginf=lo)
        w = (hi - lo) if hi > lo else 1.0
        idx = np.floor((x - lo) / w * n).astype(int)
        return np.clip(idx, 0, n - 1)

    @property
    def chi_centers(self):
        lo, hi = self.CHI_RANGE
        w = (hi - lo) / self.N_CHI_BINS
        return lo + (np.arange(self.N_CHI_BINS) + 0.5) * w

    def calibrate(self, if_vals, sf_vals, chi_vals, q=(0.5, 99.5)):

        self.IF_RANGE = tuple(np.percentile(if_vals, q))
        self.SF_RANGE = tuple(np.percentile(sf_vals, q))
        chi_vals = np.real(chi_vals)
        self.CHI_RANGE = (min(0.0, float(chi_vals.min())),
                          float(chi_vals.max()) * 1.001 + 1e-12)

    def train(self, scf, idf, gs, gd, chi):
        if_vals, sf_vals = self.pixel_features(scf, idf, gs)
        i = self._bin(if_vals, self.IF_RANGE, self.N_IF_BINS)
        s = self._bin(sf_vals, self.SF_RANGE, self.N_SF_BINS)
        k = self._bin(np.real(np.asarray(chi)).ravel(),
                      self.CHI_RANGE, self.N_CHI_BINS)
        np.add.at(self.counts, (i, s, k), 1)
        self.n_instances += 1

    def posterior(self, scf, idf, gs):
        if self.n_instances == 0:
            raise RuntimeError("Rede não treinada.")
        if_vals, sf_vals = self.pixel_features(scf, idf, gs)
        i = self._bin(if_vals, self.IF_RANGE, self.N_IF_BINS)
        s = self._bin(sf_vals, self.SF_RANGE, self.N_SF_BINS)

        c_joint = self.counts[i, s].astype(float)             # (N, n_chi)
        n_joint = c_joint.sum(axis=1)

        # Fallback hierárquico: joint -> marginal em IF -> marginal em SF
        # -> global
        c_if_all = self.counts.sum(axis=1)                    # (n_if, n_chi)
        c_sf_all = self.counts.sum(axis=0)                    # (n_sf, n_chi)
        c_glob = self.counts.sum(axis=(0, 1)).astype(float)   # (n_chi,)
        c_if = c_if_all[i].astype(float)
        c_sf = c_sf_all[s].astype(float)

        use_joint = (n_joint >= self.MIN_COUNT)[:, None]
        use_if = (c_if.sum(axis=1) >= self.MIN_COUNT)[:, None]
        use_sf = (c_sf.sum(axis=1) >= self.MIN_COUNT)[:, None]
        c = np.where(use_joint, c_joint,
            np.where(use_if, c_if,
            np.where(use_sf, c_sf, c_glob[None, :])))

        c = c + self.ALPHA
        return c / c.sum(axis=1, keepdims=True)

    def solve_with_prob(self, scf, idf, gs=None, gd=None, *args, **kwargs):

        post = self.posterior(scf, idf, gs)
        n = post.shape[0]
        res = kwargs.get("res")
        if res is None:
            side = int(round(np.sqrt(n)))
            res = (side, side)
        res = tuple(res)

        if self.DECISION == "map":
            k = post.argmax(axis=1)
            chi = self.chi_centers[k]
        else:
            chi = post @ self.chi_centers
            k = self._bin(chi, self.CHI_RANGE, self.N_CHI_BINS)
        prob = post[np.arange(n), k]
        return scf, chi.reshape(res)

    def solve(self, scf, idf, gs=None, gd=None, res=None, *args, **kwargs):
        return self.solve_with_prob(scf, idf, gs, gd, res=res, *args, **kwargs)

    def occupied_cells(self):
        return int((self.counts.sum(axis=2) > 0).sum())


    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj