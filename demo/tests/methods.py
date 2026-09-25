import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from eispy2d.core import configuration as cfg

def sum_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    N_pixels = incident_field.shape[0]


    A = np.zeros((NM * NS, N_pixels), dtype=complex)

    b = scattered_field.reshape(-1, 1, order='F')

    for s in range(NS):
        E_inc_s = incident_field[:, s:s+1]


        A_s = GS * E_inc_s.T

        A[s * NM : (s + 1) * NM, :] = A_s

    gamma = 1e-2
    A_reg = A.conj().T @ A + (gamma ** 2) * np.eye(N_pixels)
    b_reg = A.conj().T @ b

    chi_flat = np.linalg.solve(A_reg, b_reg)

    E_recover = (A @ chi_flat).reshape(NM, NS, order='F')

    return E_recover, chi_flat.reshape(recover_resolution)

def otimizar_matriz(matriz1, matriz2, matriz3, matriz4, resolucao):

    scattered = matriz1
    incident = matriz2
    GS = matriz3
    GD = matriz4

    # ============================================================
    # DIMENSÕES E ÂNGULOS
    # ============================================================

    N = GD.shape[0]
    NM, NS = scattered.shape

    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)

    # ============================================================
    # GRADE DE BUSCA LOCAL (Fixa para evitar explosão combinatória)
    # ============================================================

    cand_n = 15  # 15x15 = 225 candidatos por pixel é suficiente e rápido

    _, chi_init = sum_approximation(scattered, incident, GS, GD, resolucao)
    chi = chi_init.reshape(resolucao[0]*resolucao[1], 1)

    # Limites base para busca
    re_vals, im_vals = chi.real, chi.imag
    re_min, re_max = np.min(re_vals), np.max(re_vals)
    im_min, im_max = np.min(im_vals), np.max(im_vals)

    folga_re = max(0.5, (re_max - re_min) * 0.20)
    folga_im = max(0.5, (im_max - im_min) * 0.20)

    re = np.linspace(re_min - folga_re, re_max + folga_re, cand_n)
    im = np.linspace(im_min - folga_im, im_max + folga_im, cand_n)

    cand_flat = (re[:, None] + 1j * im[None, :]).ravel()

    #chi = np.zeros((N, 1), dtype=complex) 

    # ============================================================
    # ESTADO INICIAL EXATO
    # ============================================================

    QUANT_REP = 5

    def calcular_erro_exato(chi_vec):
        """Calcula o erro exato sem aproximações de Sherman-Morrison."""
        C_mat = np.diag(chi_vec[:, 0])
        A_mat = np.eye(N, dtype=complex) - GD @ C_mat
        E_tot = np.linalg.solve(A_mat, incident)
        pred_mat = GS @ (C_mat @ E_tot)
        
        diff_mat = scattered - pred_mat
        y_mat = np.real(diff_mat * np.conj(diff_mat))
        
        integral_phi_m = np.trapezoid(y_mat, x=phi, axis=1)
        integral_theta_m = np.trapezoid(integral_phi_m, x=theta)
        
        return np.real(np.sqrt(integral_theta_m)), E_tot, A_mat

    erro_atual, E, A = calcular_erro_exato(chi)
    print(erro_atual)

    # ============================================================
    # OTIMIZAÇÃO
    # ============================================================

    for rep in range(QUANT_REP):

        print(f"\n--- INICIANDO PASSADA {rep + 1}/{QUANT_REP} --- (Erro Inicial: {erro_atual:.6e})")

        for pos in range(N):

            # Atualiza matriz A e E no pixel atual
            C = np.diag(chi[:, 0])
            A = np.eye(N, dtype=complex) - GD @ C
            E = np.linalg.solve(A, incident)

            # ====================================================
            # SHERMAN-MORRISON (Filtro Rápido)
            # ====================================================

            q = np.linalg.solve(A, GD[:, pos])
            rrow = np.linalg.solve(A.T, np.eye(N, dtype=complex)[:, pos])
            alpha = rrow @ E

            val_atual = chi[pos, 0]
            candidates = np.append(cand_flat, val_atual)
            delta = candidates - val_atual

            denom = 1.0 - delta * rrow[pos]
            valid = np.abs(denom) > 1e-12

            beta = np.zeros_like(delta)
            beta[valid] = delta[valid] / denom[valid]

            E_cand = E[None, :, :] + beta[:, None, None] * q[None, :, None] * alpha[None, None, :]
            CE_cand = chi[None, :, :] * E_cand
            CE_cand[:, pos, :] += delta[:, None] * E_cand[:, pos, :]

            pred_cand = np.einsum('mn,kns->kms', GS, CE_cand, optimize=True)

            diff = scattered[None, :, :] - pred_cand
            y = np.real(diff * np.conj(diff))
            integral_phi = np.trapezoid(y, x=phi, axis=2)
            integral_theta = np.trapezoid(integral_phi, x=theta, axis=1)
            erros_rn = np.real(np.sqrt(integral_theta))

            # Descarta candidatos com divisão por zero no Sherman-Morrison
            erros_rn[~valid] = np.inf

            # ====================================================
            # MELHOR CANDIDATO E VALIDAÇÃO RIGOROSA DE ERRO
            # ====================================================

            best_idx = np.argmin(erros_rn)
            candidate_val = candidates[best_idx]

            # Só testa e aceita se o filtro numérico indicar melhoria
            if erros_rn[best_idx] < erro_atual:
                
                # Teste EXATO para evitar aceitar erro mascarado por precisão flutuante
                chi_teste = chi.copy()
                chi_teste[pos, 0] = candidate_val
                erro_exato_cand, _, _ = calcular_erro_exato(chi_teste)

                if erro_exato_cand < erro_atual:
                    chi[pos, 0] = candidate_val
                    erro_atual = erro_exato_cand
                    best_val = candidate_val
                    best_erro = erro_atual
                else:
                    best_val = chi[pos, 0]
                    best_erro = erro_atual
            else:
                best_val = chi[pos, 0]
                best_erro = erro_atual

            # ====================================================
            # LOG
            # ====================================================

            print(
                f"[{pos + 1}/{N}] "
                f"Melhor = {best_val.real:.4f} + {best_val.imag:.4f}j | "
                f"Residual norm error = {best_erro:.6e}"
            )

    # ============================================================
    # ERRO FINAL
    # ============================================================

    residual_error, percent_dev_er = calcula_erro(
        scattered, incident, GS, GD, resolucao, chi
    )

    print(f"\nResidual norm error Final: {residual_error:.6e}")

    E_sct = forward_solver(chi, incident, GS, GD)

    return E_sct, chi.reshape(resolucao)


def forward_solver(
    chi,
    incident_field,
    GS,
    GD
):

    N, NS = incident_field.shape

    chi_flat = chi.reshape(-1)

    C = np.diag(chi_flat)

    I = np.eye(
        N,
        dtype=complex
    )

    A_internal = I - GD @ C

    E_tot = np.linalg.solve(
        A_internal,
        incident_field
    )

    scattered_field = (
        GS @ (C @ E_tot)
    )

    return scattered_field


def calcula_erro(
    scattered_field,
    incident_field,
    GS,
    GD,
    recover_resolution,
    chi_est,
    chi_true=None
):

    N = (
        recover_resolution[0]
        * recover_resolution[1]
    )

    NM, NS = scattered_field.shape

    # ============================================================
    # CONTRASTE
    # ============================================================

    chi_flat = chi_est.reshape(-1)

    C = np.diag(chi_flat)

    I = np.eye(
        N,
        dtype=complex
    )

    # ============================================================
    # CAMPO TOTAL
    # ============================================================

    A_internal = I - GD @ C

    E_tot = np.zeros_like(
        incident_field,
        dtype=complex
    )

    for s in range(NS):

        E_tot[:, s] = np.linalg.solve(
            A_internal,
            incident_field[:, s]
        )

    # ============================================================
    # CAMPO ESPALHADO CALCULADO
    # ============================================================

    scattered_est = (
        GS
        @ (C @ E_tot)
    )

    # ============================================================
    # ÂNGULOS
    # ============================================================

    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)

    # ============================================================
    # RESIDUAL NORM ERROR
    # ============================================================

    diff = (
        scattered_field
        - scattered_est
    )

    # |Es_o - Es_a|²
    y = np.real(
        diff * np.conj(diff)
    )

    # Integração em phi
    integral_phi = np.trapezoid(
        y,
        x=phi,
        axis=1
    )

    # Integração em theta
    integral_theta = np.trapezoid(
        integral_phi,
        x=theta,
        axis=0
    )

    # Residual norm error
    residual_error = np.real(
        np.sqrt(integral_theta)
    )

    # ============================================================
    # ERRO DE PERMISSIVIDADE
    # ============================================================

    percent_dev_er = None

    if chi_true is not None:

        er_est = (
            np.real(chi_est)
            + 1.0
        )

        er_true = (
            np.real(chi_true)
            + 1.0
        )

        percent_dev_er = (
            np.mean(
                np.abs(
                    er_est - er_true
                )
                / np.abs(er_true)
            )
            * 100.0
        )

    return (
        residual_error,
        percent_dev_er
    )

def otimizar_matriz2(matriz1, matriz2, matriz3, matriz4, resolucao):
    scattered = matriz1
    incident  = matriz2
    GS        = matriz3
    GD        = matriz4

    N = GD.shape[0]
    NM, NS = scattered.shape
    theta = cfg.get_angles(NM)
    phi   = cfg.get_angles(NS)

    cand_n = 80
    QUANT_REP = 20

    es, chi_init = sum_approximation(scattered, incident, GS, GD, resolucao)
    chi = chi_init.reshape(-1, 1).copy()   # <-- agora usa chi_init, não zeros

    def calcular_erro_exato(chi_vec):
        C_mat = np.diag(chi_vec[:, 0])
        A_mat = np.eye(N, dtype=complex) - GD @ C_mat
        E_tot = np.linalg.solve(A_mat, incident)
        pred_mat = GS @ (C_mat @ E_tot)
        diff_mat = scattered - pred_mat
        y_mat = np.real(diff_mat * np.conj(diff_mat))
        ip = np.trapezoid(y_mat, x=phi, axis=1)
        it = np.trapezoid(ip, x=theta)
        return np.real(np.sqrt(it)), E_tot, A_mat

    erro_atual, E, A = calcular_erro_exato(chi)

    for rep in range(QUANT_REP):

        # ----- máscara adaptativa por passada -----
        chi_2d = chi.reshape(resolucao)
        abs_chi = np.abs(chi_2d)
        chi_max = abs_chi.max()

        thr_baixo = 0.4 * chi_max
        thr_alto  = 0.6 * chi_max

        ambiguo = (abs_chi > thr_baixo) & (abs_chi < thr_alto)

        # dilatação 8-vizinhos
        m = ambiguo.copy()
        m[1:, :]    |= ambiguo[:-1, :]
        m[:-1, :]   |= ambiguo[1:, :]
        m[:, 1:]    |= ambiguo[:, :-1]
        m[:, :-1]   |= ambiguo[:, 1:]
        m[1:, 1:]   |= ambiguo[:-1, :-1]
        m[:-1, :-1] |= ambiguo[1:, 1:]
        m[1:, :-1]  |= ambiguo[:-1, 1:]
        m[:-1, 1:]  |= ambiguo[1:, :-1]
        mask = m.ravel()

        n_ativos = mask.sum()
        print(f"\n--- PASSADA {rep+1}/{QUANT_REP} --- "
              f"(Erro: {erro_atual:.6e}, ativos: {n_ativos}/{N}, "
              f"thr_baixo={thr_baixo:.3f}, thr_alto={thr_alto:.3f})")

        # ----- grade de busca adaptativa -----
        re_min, re_max = chi.real.min(), chi.real.max()
        im_min, im_max = chi.imag.min(), chi.imag.max()
        folga_re = max(0.2, (re_max - re_min) * 0.20)
        folga_im = max(0.2, (im_max - im_min) * 0.20)
        re = np.linspace(re_min - folga_re, re_max + folga_re, cand_n)
        im = np.linspace(im_min - folga_im, im_max + folga_im, cand_n)
        cand_flat = (re[:, None] + 1j * im[None, :]).ravel()

        # ----- loop de otimização -----
        for pos in range(N):
            if not mask[pos]:
                continue

            C = np.diag(chi[:, 0])
            A = np.eye(N, dtype=complex) - GD @ C
            E = np.linalg.solve(A, incident)

            q    = np.linalg.solve(A, GD[:, pos])
            rrow = np.linalg.solve(A.T, np.eye(N, dtype=complex)[:, pos])
            alpha = rrow @ E

            val_atual = chi[pos, 0]
            candidates = np.append(cand_flat, val_atual)
            delta = candidates - val_atual

            denom = 1.0 - delta * rrow[pos]
            valid = np.abs(denom) > 1e-12

            beta = np.zeros_like(delta)
            beta[valid] = delta[valid] / denom[valid]

            E_cand  = E[None, :, :] + beta[:, None, None] * q[None, :, None] * alpha[None, None, :]
            CE_cand = chi[None, :, :] * E_cand
            CE_cand[:, pos, :] += delta[:, None] * E_cand[:, pos, :]

            pred_cand = np.einsum('mn,kns->kms', GS, CE_cand, optimize=True)

            diff = scattered[None, :, :] - pred_cand
            y = np.real(diff * np.conj(diff))
            ip = np.trapezoid(y, x=phi, axis=2)
            it = np.trapezoid(ip, x=theta, axis=1)
            erros_rn = np.real(np.sqrt(it))
            erros_rn[~valid] = np.inf

            best_idx = np.argmin(erros_rn)
            candidate_val = candidates[best_idx]

            if erros_rn[best_idx] < erro_atual:
                chi_teste = chi.copy()
                chi_teste[pos, 0] = candidate_val
                erro_exato_cand, _, _ = calcular_erro_exato(chi_teste)
                if erro_exato_cand < erro_atual:
                    chi[pos, 0] = candidate_val
                    erro_atual = erro_exato_cand

    residual_error, _ = calcula_erro(scattered, incident, GS, GD, resolucao, chi)
    print(f"\nResidual norm error Final: {residual_error:.6e}")

    E_sct = forward_solver(chi, incident, GS, GD)
    return E_sct, chi.reshape(resolucao)

from eispy2d.api import api

params = {'resolution':(30,30), 'disp':True}
api.evaluate(otimizar_matriz, params)