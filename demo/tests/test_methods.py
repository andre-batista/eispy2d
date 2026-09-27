import time
import pickle
import numpy as np
from functools import partial

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api

import methods

params = {'resolution': (21, 21)}

cand_n_values   = list(range(10, 80, 5))     # 5, 10, ..., 100
quant_rep_values = list(range(2, 24, 4))     # 2, 4, ..., 20

# Arquivo de saída
output_file = 'varredura_candn_quantrep.pkl'


resultados = []

for quant_rep in quant_rep_values:
    for cand_n in cand_n_values:

        alg = partial(methods.otimizar_matriz3, cand_n=cand_n, quant_rep=quant_rep)
        alg.__name__ = methods.otimizar_matriz3.__name__

        t0 = time.perf_counter()
        try:
            r = api.evaluate(alg, params)
            t_exec = time.perf_counter() - t0

            zeta_epad = float(r.zeta_epad) 
            zeta_rn   = float(r.zeta_rn)  

            resultados.append({
                'cand_n':    cand_n,
                'quant_rep': quant_rep,
                'zeta_epad': zeta_epad,
                'zeta_rn':   zeta_rn,
                'tempo':     t_exec,
                'sucesso':   True,
            })

        except Exception as e:
            t_exec = time.perf_counter() - t0
            resultados.append({
                'cand_n':    cand_n,
                'quant_rep': quant_rep,
                'zeta_epad': None,
                'zeta_rn':   None,
                'tempo':     t_exec,
                'sucesso':   False,
                'erro':      repr(e),
            })


with open(output_file, 'wb') as f:
    pickle.dump({
        'params':           params,
        'cand_n_values':    cand_n_values,
        'quant_rep_values': quant_rep_values,
        'resultados':       resultados,
    }, f)