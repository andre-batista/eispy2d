import sys
import os

from run import instance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))

from eispy2d.api import api

import numpy as np
import math
import pickle

class Train:

    def __init__(self):
        self.chain = {}

    # ============================================================
    # SAVE / READ
    # ============================================================
    def save(self, filename="chain_data.pkl"):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.chain, f)
            print(f"Dados salvos em {filename} ({len(self.chain)} chaves).")
        except Exception as e:
            print(f"Erro ao salvar: {e}")

    def read(self, filename="chain_data.pkl"):
        """
        Adiciona ao chain existente. Se a chave já existir, faz média.
        """
        try:
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)

            for key, value in loaded.items():
                if key in self.chain:
                    self.chain[key] = (self.chain[key] + value) / 2.0
                else:
                    self.chain[key] = value

            print(f"Dados lidos de {filename}. Total de chaves: {len(self.chain)}.")
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado.")
        except Exception as e:
            print(f"Erro ao ler: {e}")

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self, scf, idf, gs, gd, chi):
        """
        scf : (nm, ns)  complex
        idf : (N,  ns)  complex
        gs  : (nm, N)   complex
        gd  : (N,  N)   complex
        chi : (nx, ny)  complex
        N = nx * ny
        """
        nx, ny = chi.shape
        N = nx * ny
        nm, ns = scf.shape

        # Sanidade: idf deve ter N linhas
        assert idf.shape[0] == N, \
            f"idf tem {idf.shape[0]} linhas, esperado N={N}"

        # Pré-computa a média de cada coluna de scf: shape (ns,)
        scf_means = np.mean(scf, axis=0)   # média sobre nm

        # Pré-computa a média de cada linha de idf: shape (N,)
        idf_means = np.mean(idf, axis=1)   # média sobre ns

        for px in range(nx):
            for py in range(ny):
                idx = px * ny + py
                incident_mean = idf_means[idx]      # escalar (complex)

                for ps in range(ns):
                    scattered_mean = scf_means[ps]  # escalar (complex)

                    key = (complex(incident_mean), complex(scattered_mean))

                    if key in self.chain:
                        self.chain[key] = (self.chain[key] + chi[px, py]) / 2.0
                    else:
                        self.chain[key] = complex(chi[px, py])

    # ============================================================
    # SOLVE
    # ============================================================
    def solve_old(self, scf, idf, gs, gd, res):
        """
        Reconstrói chi a partir do chain.
        res = (nx, ny)
        """
        nx, ny = res
        N = nx * ny
        nm, ns = scf.shape

        assert idf.shape[0] == N, \
            f"idf tem {idf.shape[0]} linhas, esperado N={N}"

        chi = np.zeros((nx, ny), dtype=complex)

        # Pré-computa médias
        scf_means = np.mean(scf, axis=0)   # (ns,)
        idf_means = np.mean(idf, axis=1)   # (N,)

        # Extrai chaves/valores para arrays NumPy (vetorização)
        keys    = list(self.chain.keys())
        values  = np.array([self.chain[k] for k in keys], dtype=complex)
        inc_arr = np.array([k[0] for k in keys], dtype=complex)
        sca_arr = np.array([k[1] for k in keys], dtype=complex)

        for px in range(nx):
            for py in range(ny):
                idx = px * ny + py
                incident_mean = idf_means[idx]

                pchi_total = 0 + 0j

                for ps in range(ns):
                    scattered_mean = scf_means[ps]

                    # Distâncias vetorizadas (partes real e imaginária)
                    di = np.sqrt((inc_arr.real - incident_mean.real) ** 2 +
                                 (inc_arr.imag - incident_mean.imag) ** 2)
                    ds = np.sqrt((sca_arr.real - scattered_mean.real) ** 2 +
                                 (sca_arr.imag - scattered_mean.imag) ** 2)

                    prod = di * ds
                    mask = prod > 0
                    if np.any(mask):
                        pchi = np.sum(values[mask] / prod[mask]) * np.sum(prod[mask])
                    else:
                        pchi = 0 + 0j

                    pchi_total += pchi

                chi[px, py] = pchi_total / ns if ns > 0 else 0 + 0j

        return scf, chi

    def get_pchi(self, s_p, i_p):
        pchi = 0 + 0j
        dtotal = 0
        for key in self.chain.keys():
            di = self.distance(i_p, key[0])
            ds = self.distance(s_p, key[1])

            d = di * ds
            
            if d == 0:
                return self.chain[key]

            dtotal += d 

            pchi += self.chain[key] / d

        pchi /= dtotal

        return pchi

    
    def solve(self, scf, icf, gs, gd, res):
        
        nm, ns = scf.shape
        nx, ny = res
        N = nx * ny

        scf_means = np.mean(scf, axis=0)   # (ns,)
        icf_means = np.mean(icf, axis=1)   # (N,)
        
        chi = np.zeros((nx, ny), dtype=complex)

        print(icf)
        print(scf)
        print(icf_means)
        print(scf_means)


        for px in range(0, nx):
            
            for py in range(0, ny):
                p = px * nx + py
                icf_m = icf_means[p]
                scf_mp = 0 + 0j 
                for ps in range(0, ns):
                    scf_mp += scf_means[ps]

                scf_mp /= ns
                pchi = self.get_pchi(icf_m, scf_mp)

                chi[px, py] = pchi
                

        return scf, chi
            


    # ============================================================
    # DISTANCE
    # ============================================================
    @staticmethod
    def distance(x, y):
        """Distância euclidiana entre dois complexos (aceita escalar ou array)."""
        x = np.asarray(x, dtype=complex)
        y = np.asarray(y, dtype=complex)
        rd  = y.real - x.real
        imd = y.imag - x.imag
        return np.sqrt(rd ** 2 + imd ** 2)


# Exemplo de uso:

# Criando uma instância
trainer = Train()

# Exemplo de dados (substitua pelos seus dados reais)

params = {"resolution":(5, 5),
          "number_measurements":2,
          "number_sources":2,
          "disp":True}

scf, idf, gs, gd, res, chi = instance(params)


# chi.shape -> resolution * 2/3

# Treinando
trainer.train(scf, idf, gs, gd, chi)

print("Real chi")
print(chi)
print("===")

c_r = trainer.solve(scf, idf, gs, gd, res)
print("Chi recon")
print(c_r)

# Salvando
#trainer.save("dados_treino.pkl")

# Lendo em uma nova instância
#trainer2 = Train()
#trainer2.read("dados_treino.pkl")

api.evaluate(trainer.solve, params)

"""
params = {"resolution":(15, 14),
          "number_measurements":12,
          "number_sources":13}

api.evaluate(t, params)

"""
