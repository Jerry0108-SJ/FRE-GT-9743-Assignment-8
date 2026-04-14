from enum import Enum
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.stats import norm
from fixedincomelib.analytics.european_options import (
    CallOrPut,
    SimpleMetrics,
    EuropeanOptionAnalytics,
)


class SabrMetrics(Enum):

    # parameters
    ALPHA = "alpha"
    BETA = "beta"
    NU = "nu"
    RHO = "rho"

    # risk
    DALPHA = "dalpha"
    DLNSIGMA = "dlnsigma"
    DNORMALSIGMA = "dnormalsigma"
    DBETA = "dbeta"
    DRHO = "drho"
    DNU = "dnu"
    DFORWARD = "dforward"
    DSTRIKE = "dstrike"
    DTTE = "dtte"
    DSTRIKESTRIKE = "dstrikestrike"

    # (alpha, beta, nu, rho, forward, strike, tte) => \sigma_k
    D_LN_SIGMA_D_FORWARD = "d_ln_sigma_d_forward"
    D_LN_SIGMA_D_STRIKE = "d_ln_sigma_d_strike"
    D_LN_SIGMA_D_TTE = "d_ln_sigma_d_tte"
    D_LN_SIGMA_D_ALPHA = "d_ln_sigma_d_alpha"
    D_LN_SIGMA_D_BETA = "d_ln_sigma_d_beta"
    D_LN_SIGMA_D_NU = "d_ln_sigma_d_nu"
    D_LN_SIGMA_D_RHO = "d_ln_sigma_d_rho"
    D_LN_SIGMA_D_STRIKESTRIKE = "d_ln_sigma_d_strike_strike"

    # (\sigma_ln_atm, f, tte, beta, nu, rho) => alpha
    D_ALPHA_D_LN_SIGMA_ATM = "d_alpha_d_ln_sigma_atm"
    D_ALPHA_D_FORWARD = "d_alpha_d_forward"
    D_ALPHA_D_TTE = "d_alpha_d_tte"
    D_ALPHA_D_BETA = "d_alpha_d_beta"
    D_ALPHA_D_NU = "d_alpha_d_nu"
    D_ALPHA_D_RHO = "d_alpha_d_rho"

    # (alpha, beta, nu, rho, f, tte) => \sigma_n_atm
    D_NORMAL_SIGMA_D_ALPHA = "d_normal_sigma_d_alpha"
    D_NORMAL_SIGMA_D_BETA = "d_normal_sigma_d_beta"
    D_NORMAL_SIGMA_D_NU = "d_normal_sigma_d_nu"
    D_NORMAL_SIGMA_D_RHO = "d_normal_sigma_d_rho"
    D_NORMAL_SIGMA_D_FORWARD = "d_normal_sigma_d_forward"
    D_NORMAL_SIGMA_D_TTE = "d_normal_sigma_d_tte"
    D_ALPHA_D_NORMAL_SIGMA_ATM = "d_alpha_d_normal_sigma_atm"

    @classmethod
    def from_string(cls, value: str) -> "SabrMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(f"Invalid token: {value}") from e

    def to_string(self) -> str:
        return self.value


class SABRAnalytics:

    EPSILON = 1e-6

    ### parameters conversion

    # solver to back out lognormal vol from alpha and sensitivities
    # please implement the _vol_and_risk function to make this work
    @staticmethod
    def lognormal_vol_from_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SabrMetrics | SimpleMetrics, float]:

        res: Dict[Any, float] = {}

        ln_sigma, risks = SABRAnalytics._vol_and_risk(
            forward + shift, strike + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma

        if len(risks) == 0:
            return res

        res.update(risks)
        return res

    @staticmethod
    def alpha_from_atm_lognormal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_lognormal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
        max_iter: Optional[int] = 50,
        tol: Optional[float] = 1e-12,
    ) -> Dict[SabrMetrics, float]:

        if forward + shift <= 0.0:
            raise ValueError("forward must be > 0")
        if time_to_expiry < 0.0:
            raise ValueError("time_to_expiry must be >= 0")
        if sigma_atm_lognormal <= 0.0:
            raise ValueError("sigma_atm_lognormal must be > 0")
        if abs(rho) >= 1.0:
            raise ValueError("rho must be in (-1,1)")
        if nu < 0.0:
            raise ValueError("nu must be >= 0")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta should be in [0,1] for standard SABR usage")

        # newton + bisec fallback
        # root finding
        # f = F(alpha, theta) - ln_sigma = 0
        # where F is lognormal_vol_from_alpha
        # alpha^* = alpha(ln_sigma, theta)

         # Newton iteration: f(alpha) = sigma_atm_ln(alpha) - sigma_atm_lognormal = 0
        shift = shift or 0.0
        tol = tol or 1e-12
        max_iter_n = int(max_iter) if max_iter is not None else 50
        this_res = None
        alpha = sigma_atm_lognormal * (forward + shift) ** (1.0 - beta)
        for _ in range(max_iter_n):
            sigma_ln, ln_risk = SABRAnalytics._vol_and_risk(
                forward + shift, forward + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk=True
            )
            f = sigma_ln - sigma_atm_lognormal
            # check convergence BEFORE update
            if abs(f) < tol:
                this_res = (sigma_ln, ln_risk)
                break
            # df/dalpha = sigma_ln * D_LN_SIGMA_D_ALPHA
            df_dalpha = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            alpha -= f / df_dalpha
        else:
            raise RuntimeError("alpha_from_atm_lognormal_sigma: Newton did not converge")

        res: Dict[SabrMetrics, float] = {SabrMetrics.ALPHA: alpha}

        if calc_risk:
            # IFT: F(alpha(theta), theta) = sigma_target
            # df/dalpha * dalpha/dln_sigma = 1  =>  dalpha/dln_sigma = 1/df_dalpha
            # df/dalpha * dalpha/dtheta + df/dtheta = 0  =>  dalpha/dtheta = -df/dtheta / df_dalpha
            sigma_ln, ln_risk = this_res
            df_dalpha = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_ALPHA]

            res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM] = 1.0 / df_dalpha

            # for ATM forward sensitivity, K=F moves together with F
            df_dF = sigma_ln * (
                ln_risk[SabrMetrics.D_LN_SIGMA_D_FORWARD] + ln_risk[SabrMetrics.D_LN_SIGMA_D_STRIKE]
            )
            res[SabrMetrics.D_ALPHA_D_FORWARD] = -df_dF / df_dalpha

            df_dT = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_TTE]
            res[SabrMetrics.D_ALPHA_D_TTE] = -df_dT / df_dalpha

            df_db = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_BETA]
            res[SabrMetrics.D_ALPHA_D_BETA] = -df_db / df_dalpha

            df_dr = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_RHO]
            res[SabrMetrics.D_ALPHA_D_RHO] = -df_dr / df_dalpha

            df_dn = sigma_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_NU]
            res[SabrMetrics.D_ALPHA_D_NU] = -df_dn / df_dalpha

        return res

    # conversion to alpha from normal atm vol
    @staticmethod
    def alpha_from_atm_normal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_normal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: bool = False,
        max_iter: int = 50,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SabrMetrics, float]:

        shift = shift or 0.0
        F_shifted = forward + shift

        # step 1: convert normal ATM vol -> lognormal ATM vol at F=K (ATM)
        conv_res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
            F_shifted, F_shifted, time_to_expiry, sigma_atm_normal, calc_risk=calc_risk
        )
        sigma_atm_ln = conv_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]

        # step 2: back out alpha from lognormal ATM vol
        ln_sigma_res = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward, time_to_expiry, sigma_atm_ln, beta, rho, nu, shift, calc_risk, max_iter, tol
        )

        final_res: Dict[SabrMetrics, float] = {SabrMetrics.ALPHA: ln_sigma_res[SabrMetrics.ALPHA]}

        if calc_risk:
            # chain rule: dalpha/d(normal_sigma) = dalpha/d(ln_sigma) * d(ln_sigma)/d(normal_sigma)
            d_alpha_d_lnsig = ln_sigma_res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            d_lnsig_d_nsig = conv_res[SimpleMetrics.D_LN_VOL_D_N_VOL]

            final_res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM] = d_alpha_d_lnsig * d_lnsig_d_nsig

            # dalpha/dF: sigma_ln changes when F changes (ATM, K=F), plus direct alpha(sigma_ln,F) sensitivity
            final_res[SabrMetrics.D_ALPHA_D_FORWARD] = (
                d_alpha_d_lnsig * (
                    conv_res[SimpleMetrics.D_LN_VOL_D_FORWARD] + conv_res[SimpleMetrics.D_LN_VOL_D_STRIKE]
                )
                + ln_sigma_res[SabrMetrics.D_ALPHA_D_FORWARD]
            )

            # dalpha/dT: sigma_ln changes with T, plus direct alpha(sigma_ln,T) sensitivity
            final_res[SabrMetrics.D_ALPHA_D_TTE] = (
                d_alpha_d_lnsig * conv_res[SimpleMetrics.D_LN_VOL_D_TTE]
                + ln_sigma_res[SabrMetrics.D_ALPHA_D_TTE]
            )

            # beta/rho/nu only appear in SABR (not in vol conversion)
            final_res[SabrMetrics.D_ALPHA_D_BETA] = ln_sigma_res[SabrMetrics.D_ALPHA_D_BETA]
            final_res[SabrMetrics.D_ALPHA_D_RHO] = ln_sigma_res[SabrMetrics.D_ALPHA_D_RHO]
            final_res[SabrMetrics.D_ALPHA_D_NU] = ln_sigma_res[SabrMetrics.D_ALPHA_D_NU]

        return final_res

    # compute normal ATM vol from alpha and SABR params
    @staticmethod
    def atm_normal_sigma_from_alpha(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: bool = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[Any, float]:

        shift = shift or 0.0
        F_shifted = forward + shift

        # step 1: lognormal ATM vol from alpha (F=K at ATM)
        sigma_atm_ln, ln_risk = SABRAnalytics._vol_and_risk(
            F_shifted, F_shifted, time_to_expiry, alpha, beta, rho, nu, calc_risk=calc_risk
        )

        # step 2: convert lognormal ATM vol -> normal ATM vol
        conv_res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(
            F_shifted, F_shifted, time_to_expiry, sigma_atm_ln, calc_risk=calc_risk, tol=tol
        )
        sigma_atm_n = conv_res[SimpleMetrics.IMPLIED_NORMAL_VOL]

        res: Dict[Any, float] = {SimpleMetrics.IMPLIED_NORMAL_VOL: sigma_atm_n}

        if calc_risk:
            d_nv_d_lnv = conv_res[SimpleMetrics.D_N_VOL_D_LN_VOL]

            # dσ_N/dα = d_nv_d_lnv * σ_LN * D_LN_SIGMA_D_ALPHA
            res[SabrMetrics.D_NORMAL_SIGMA_D_ALPHA] = (
                d_nv_d_lnv * sigma_atm_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            )
            # inverse: dα/dσ_N_ATM = 1 / dσ_N/dα
            res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM] = 1.0 / res[SabrMetrics.D_NORMAL_SIGMA_D_ALPHA]

            res[SabrMetrics.D_NORMAL_SIGMA_D_BETA] = (
                d_nv_d_lnv * sigma_atm_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_BETA]
            )
            res[SabrMetrics.D_NORMAL_SIGMA_D_RHO] = (
                d_nv_d_lnv * sigma_atm_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_RHO]
            )
            res[SabrMetrics.D_NORMAL_SIGMA_D_NU] = (
                d_nv_d_lnv * sigma_atm_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_NU]
            )

            # dσ_N/dF (ATM, K=F moves with F): direct conv F+K sensitivity + through σ_LN
            res[SabrMetrics.D_NORMAL_SIGMA_D_FORWARD] = (
                conv_res[SimpleMetrics.D_N_VOL_D_FORWARD] + conv_res[SimpleMetrics.D_N_VOL_D_STRIKE]
                + d_nv_d_lnv * sigma_atm_ln * (
                    ln_risk[SabrMetrics.D_LN_SIGMA_D_FORWARD] + ln_risk[SabrMetrics.D_LN_SIGMA_D_STRIKE]
                )
            )

            # dσ_N/dT: direct conv T sensitivity + through σ_LN
            res[SabrMetrics.D_NORMAL_SIGMA_D_TTE] = (
                conv_res[SimpleMetrics.D_N_VOL_D_TTE]
                + d_nv_d_lnv * sigma_atm_ln * ln_risk[SabrMetrics.D_LN_SIGMA_D_TTE]
            )

        return res

    ### option pricing

    @staticmethod
    def european_option_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        ln_sigma_and_sensitivities = SABRAnalytics.lognormal_vol_from_alpha(
            forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk
        )
        ln_iv = ln_sigma_and_sensitivities[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        value_and_sensitivities = EuropeanOptionAnalytics.european_option_log_normal(
            forward + shift, strike + shift, time_to_expiry, ln_iv, opt_type, calc_risk
        )

        ### risk(analytic)
        if calc_risk:
            ## first order risks
            dvdsigma = value_and_sensitivities[SimpleMetrics.VEGA]
            value_and_sensitivities.pop(SimpleMetrics.VEGA)
            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_TTE]
            )
            # sabr alpha/beta/nu/rho
            for key, risk in [
                (SabrMetrics.DALPHA, SabrMetrics.D_LN_SIGMA_D_ALPHA),
                (SabrMetrics.DBETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_LN_SIGMA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_LN_SIGMA_D_NU),
            ]:
                value_and_sensitivities[key] = dvdsigma * ln_sigma_and_sensitivities[risk]
            # strike
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_STRIKE]
            )

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]
            # strike
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike + SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift + SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]

            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike - SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift - SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK_2] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

            # gamma
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward + SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift + SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward - SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift - SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # Given function
    @staticmethod
    def european_option_ln_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        ln_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift, calc_risk
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        value_and_sensitivities = SABRAnalytics.european_option_alpha(
            forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk
        )

        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )
            # ln_sigma
            value_and_sensitivities[SabrMetrics.DLNSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            )
            # sabr beta/rho/nu
            for key, risk in [
                (SabrMetrics.DBETA, SabrMetrics.D_ALPHA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_ALPHA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_ALPHA_D_NU),
            ]:
                value_and_sensitivities[key] += dvdalpha * alpha_and_sensitivities[risk]

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward + SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_up,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward - SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_dn,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # European call/put SABR risk with normal vol input, please implement this function with european_option_alpha api
    @staticmethod
    def european_option_normal_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        normal_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):
        """
        Please implement this function with european_option_alpha api

        """
        ### pv
        _calc_risk: bool = bool(calc_risk)
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_normal_sigma(
            forward, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift, _calc_risk
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        value_and_sensitivities: Dict[Any, float] = SABRAnalytics.european_option_alpha(
            forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, _calc_risk
        )

        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )
            # normal sigma vega
            value_and_sensitivities[SabrMetrics.DNORMALSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM]
            )
            # sabr beta/rho/nu
            for key, risk in [
                (SabrMetrics.DBETA, SabrMetrics.D_ALPHA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_ALPHA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_ALPHA_D_NU),
            ]:
                value_and_sensitivities[key] += dvdalpha * alpha_and_sensitivities[risk]

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward + SABRAnalytics.EPSILON, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON, strike, time_to_expiry, opt_type,
                alpha_up, beta, rho, nu, shift,
            )[SimpleMetrics.PV]

            res_dn = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward - SABRAnalytics.EPSILON, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON, strike, time_to_expiry, opt_type,
                alpha_dn, beta, rho, nu, shift,
            )[SimpleMetrics.PV]

            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON ** 2
            )

        return value_and_sensitivities

    @staticmethod
    def pdf_and_cdf(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        strikes: List[float],
        shift: Optional[float] = 0.0,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Compute the SABR-implied PDF and CDF over a grid of strikes using
        Breeden-Litzenberger: pdf(K) = d²C/dK², cdf(K) = 1 + dC/dK.

        Uses the analytical formula:
            C(K) = (F - K) * N(d) + sigma_n * sqrt(T) * n(d)   [normal model at each K]
        where sigma_n is the SABR-implied normal vol at strike K.

        Returns (xs, xs_shifted, cdf, pdf) as plain lists.
        """
        xs = list(strikes)
        _shift = shift if shift is not None else 0.0
        xs_shifted = [K + _shift for K in xs]

        # compute call prices on the shifted grid
        pvs = []
        for K in xs:
            try:
                pv = SABRAnalytics.european_option_alpha(
                    forward, K, time_to_expiry, CallOrPut.CALL, alpha, beta, rho, nu, shift
                )[SimpleMetrics.PV]
            except Exception:
                pv = max(forward - K, 0.0)
            pvs.append(pv)

        n = len(xs)
        cdf_vals = [0.0] * n
        pdf_vals = [0.0] * n

        for i in range(n):
            if i == 0 or i == n - 1:
                # boundary: use one-sided finite difference
                if i == 0:
                    dK = xs[1] - xs[0]
                    dc_dk = (pvs[1] - pvs[0]) / dK
                else:
                    dK = xs[-1] - xs[-2]
                    dc_dk = (pvs[-1] - pvs[-2]) / dK
                cdf_vals[i] = 1.0 + dc_dk
                pdf_vals[i] = 0.0
            else:
                dK_avg = (xs[i + 1] - xs[i - 1]) / 2.0
                # first derivative (central difference)
                dc_dk = (pvs[i + 1] - pvs[i - 1]) / (xs[i + 1] - xs[i - 1])
                cdf_vals[i] = 1.0 + dc_dk
                # second derivative
                d2c_dk2 = (pvs[i + 1] - 2 * pvs[i] + pvs[i - 1]) / (dK_avg ** 2)
                pdf_vals[i] = d2c_dk2

        return xs, xs_shifted, cdf_vals, pdf_vals

    ### helpers

    @staticmethod
    def w2_risk(F, K, T, a, b, r, n) -> Dict:

        risk = {}

        risk[SabrMetrics.DALPHA] = (1 - b) ** 2 / 12 * a / (F * K) ** (1 - b) + b * r * n / (
            4 * (F * K) ** ((1 - b) / 2)
        )
        risk[SabrMetrics.DBETA] = (
            1 / 12 * (b - 1) * a**2 * (F * K) ** (b - 1)
            + 1 / 24 * (b - 1) ** 2 * a**2 * (F * K) ** (b - 1) * np.log(F * K)
            + 1 / 4 * a * r * n * (F * K) ** ((b - 1) / 2)
            + 1 / 8 * a * b * r * n * (F * K) ** ((b - 1) / 2) * np.log(F * K)
        )
        risk[SabrMetrics.DRHO] = 1 / 4 * a * b * n * (F * K) ** ((b - 1) / 2) - 1 / 4 * n**2 * r
        risk[SabrMetrics.DNU] = (
            1 / 4 * a * b * r * (F * K) ** ((b - 1) / 2) + 1 / 6 * n - 1 / 4 * r**2 * n
        )
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 3 / 24 * a**2 * (F * K) ** (
            b - 2
        ) * K + a * r * n * b * (b - 1) / 8 * K ** ((b - 1) / 2) * F ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKE] = (b - 1) ** 3 / 24 * a**2 * F ** (b - 1) * K ** (
            b - 2
        ) + a * b * r * n * (b - 1) / 8 * F ** ((b - 1) / 2) * K ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKESTRIKE] = (b - 1) ** 3 / 24 * a**2 * (b - 2) * F ** (
            b - 1
        ) * K ** (b - 3) + a * b * r * n / 16 * (b - 1) * (b - 3) * F ** ((b - 1) / 2) * K ** (
            (b - 5) / 2
        )

        return risk

    @staticmethod
    def w1_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)

        risk = {}
        risk[SabrMetrics.DALPHA] = 0.0
        risk[SabrMetrics.DBETA] = (b - 1) / 12.0 * log_FK**2 + (b - 1) ** 3 / 480 * log_FK**4
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 0.0
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 2 / 12 * log_FK / F + (
            b - 1
        ) ** 4 / 480 / F * log_FK**3
        risk[SabrMetrics.DSTRIKE] = (
            -((b - 1) ** 2) / 12 * log_FK / K - (b - 1) ** 4 / 480 / K * log_FK**3
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            (b - 1) ** 2 / 12 / K**2
            + (b - 1) ** 2 / 12 * log_FK / K**2
            + (b - 1) ** 4 / 160 * log_FK**2 / K**2
            + (b - 1) ** 4 / 480 * log_FK**3 / K**2
        )

        return risk

    @staticmethod
    def z_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        # z = n / a * log_FK * fk

        risk = {}
        risk[SabrMetrics.DALPHA] = -n / a * log_FK * fk / a
        risk[SabrMetrics.DBETA] = -1.0 / 2 * n / a * log_FK * fk * np.log(F * K)
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 1.0 / a * log_FK * fk
        risk[SabrMetrics.DFORWARD] = (
            n * (1 - b) * K / 2 / a * (F * K) ** ((-b - 1) / 2) * log_FK + n / a * fk / F
        )
        risk[SabrMetrics.DSTRIKE] = (
            n * F * (1 - b) / 2 / a * log_FK * (F * K) ** ((-b - 1) / 2) - n / a * fk / K
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            n / a * F ** ((1 - b) / 2) * K ** ((-b - 3) / 2) * (log_FK * (b**2 - 1) / 4 + b)
        )

        return risk

    @staticmethod
    def x_risk(F, K, T, a, b, r, n) -> Dict:

        logFK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        z = n / a * fk * logFK
        dx_dz = 1 / np.sqrt(1 - 2 * r * z + z**2)

        risk = {}
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dx_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dx_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DRHO] = 1 / (1 - r) + (-z * dx_dz - 1) / (1 / dx_dz + z - r)
        risk[SabrMetrics.DNU] = dx_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dx_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dx_dz * risk_z[SabrMetrics.DSTRIKE]

        risk[SabrMetrics.DSTRIKESTRIKE] = (r - z) * dx_dz**3 * (
            risk_z[SabrMetrics.DSTRIKE] ** 2
        ) + dx_dz * risk_z[SabrMetrics.DSTRIKESTRIKE]

        return risk

    @staticmethod
    def C_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)

        z = n / a * log_FK * fk
        risk = {}

        C0 = 1.0
        C1 = -r / 2.0
        C2 = -(r**2) / 4.0 + 1.0 / 6.0
        C3 = -(1.0 / 4.0 * r**2 - 5.0 / 24.0) * r
        C4 = -5.0 / 16.0 * r**4 + 1.0 / 3.0 * r**2 - 17.0 / 360.0
        C5 = -(7.0 / 16.0 * r**4 - 55.0 / 96.0 * r**2 + 37.0 / 240.0) * r

        dC_dz = C1 + 2 * C2 * z + 3 * C3 * z**2 + 4 * C4 * z**3 + 5 * C5 * z**4
        dC2_dz2 = 2 * C2 + 6 * C3 * z + 12 * C4 * z**2 + 20 * C5 * z**3

        risk[SabrMetrics.DRHO] = (
            -1.0 / 2 * z
            + 5.0 / 24 * z**3
            - 37.0 / 240 * z**5
            - 1.0 / 2 * z**2 * r
            + 2.0 / 3 * z**4 * r
            - 3.0 / 4 * z**3 * r**2
            + 55.0 / 32 * z**5 * r**2
            - 5.0 / 4 * z**4 * r**3
            - 35.0 / 16 * z**5 * r**4
        )
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dC_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dC_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DNU] = dC_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dC_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dC_dz * risk_z[SabrMetrics.DSTRIKE]
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            dC_dz * risk_z[SabrMetrics.DSTRIKESTRIKE] + dC2_dz2 * risk_z[SabrMetrics.DSTRIKE] ** 2
        )
        return risk

    @staticmethod
    def _vol_and_risk(
        F, K, T, a, b, r, n, calc_risk=False, z_cut=1e-2
    ) -> Tuple[float, Dict[SabrMetrics, float]]:
        """
        Get analytical solution Lognormal Vol and Greeks
        """

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        greeks: Dict[SabrMetrics, float] = {}

        z = n / a * log_FK * fk

        # log-moneyness correction denominator
        w1 = 1.0 + (1 - b) ** 2 / 24.0 * log_FK ** 2 + (1 - b) ** 4 / 1920.0 * log_FK ** 4

        # time correction
        w2 = (
            (1 - b) ** 2 / 24.0 * a ** 2 / fk ** 2
            + b * r * n * a / (4.0 * fk)
            + (2.0 - 3.0 * r ** 2) / 24.0 * n ** 2
        )

        if abs(z) < z_cut:
            # expansion when z is small: use Taylor series C(z) ≈ z / x(z)
            C0 = 1.0
            C1 = -r / 2.0
            C2 = -(r ** 2) / 4.0 + 1.0 / 6.0
            C3 = -(1.0 / 4.0 * r ** 2 - 5.0 / 24.0) * r
            C4 = -5.0 / 16.0 * r ** 4 + 1.0 / 3.0 * r ** 2 - 17.0 / 360.0
            C5 = -(7.0 / 16.0 * r ** 4 - 55.0 / 96.0 * r ** 2 + 37.0 / 240.0) * r

            C_val = C0 + C1 * z + C2 * z ** 2 + C3 * z ** 3 + C4 * z ** 4 + C5 * z ** 5

            sigma = a / (fk * w1) * C_val * (1.0 + w2 * T)

            if calc_risk:
                risk_w1 = SABRAnalytics.w1_risk(F, K, T, a, b, r, n)
                risk_w2 = SABRAnalytics.w2_risk(F, K, T, a, b, r, n)
                risk_C = SABRAnalytics.C_risk(F, K, T, a, b, r, n)

                factor_w2 = T / (1.0 + w2 * T)

                # d(ln fk)/d(param): fk = (FK)^((1-b)/2)
                dln_fk_dF = (1.0 - b) / (2.0 * F)
                dln_fk_dK = (1.0 - b) / (2.0 * K)
                dln_fk_db = -0.5 * np.log(F * K)

                # d(ln sigma)/d(param) = [1/a for alpha] + dC/C - d(ln fk) - dw1/w1 + T*dw2/(1+w2T)
                greeks[SabrMetrics.D_LN_SIGMA_D_ALPHA] = (
                    1.0 / a
                    + risk_C[SabrMetrics.DALPHA] / C_val
                    - risk_w1[SabrMetrics.DALPHA] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DALPHA]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_BETA] = (
                    risk_C[SabrMetrics.DBETA] / C_val
                    - dln_fk_db
                    - risk_w1[SabrMetrics.DBETA] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DBETA]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_RHO] = (
                    risk_C[SabrMetrics.DRHO] / C_val
                    - risk_w1[SabrMetrics.DRHO] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DRHO]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_NU] = (
                    risk_C[SabrMetrics.DNU] / C_val
                    - risk_w1[SabrMetrics.DNU] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DNU]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_FORWARD] = (
                    risk_C[SabrMetrics.DFORWARD] / C_val
                    - dln_fk_dF
                    - risk_w1[SabrMetrics.DFORWARD] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DFORWARD]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKE] = (
                    risk_C[SabrMetrics.DSTRIKE] / C_val
                    - dln_fk_dK
                    - risk_w1[SabrMetrics.DSTRIKE] / w1
                    + factor_w2 * risk_w2[SabrMetrics.DSTRIKE]
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = w2 / (1.0 + w2 * T)

                # d²(ln sigma)/dK²:
                # d²(ln C)/dK² + (1-b)/(2K²) - d²(ln w1)/dK² + T*d²w2/dK²/(1+w2T) - T²*(dw2/dK)²/(1+w2T)²
                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = (
                    (risk_C[SabrMetrics.DSTRIKESTRIKE] * C_val - risk_C[SabrMetrics.DSTRIKE] ** 2) / C_val ** 2
                    + (1.0 - b) / (2.0 * K ** 2)
                    - (risk_w1[SabrMetrics.DSTRIKESTRIKE] * w1 - risk_w1[SabrMetrics.DSTRIKE] ** 2) / w1 ** 2
                    + factor_w2 * risk_w2[SabrMetrics.DSTRIKESTRIKE]
                    - factor_w2 ** 2 * risk_w2[SabrMetrics.DSTRIKE] ** 2
                )

            return sigma, greeks

        # raw SABR (|z| >= z_cut)
        x_val = np.log((np.sqrt(1.0 - 2.0 * r * z + z ** 2) + z - r) / (1.0 - r))

        sigma = a / (fk * w1) * z / x_val * (1.0 + w2 * T)

        if calc_risk:
            risk_w1 = SABRAnalytics.w1_risk(F, K, T, a, b, r, n)
            risk_w2 = SABRAnalytics.w2_risk(F, K, T, a, b, r, n)
            risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)
            risk_x = SABRAnalytics.x_risk(F, K, T, a, b, r, n)

            factor_w2 = T / (1.0 + w2 * T)

            # d(ln fk)/d(param)
            dln_fk_dF = (1.0 - b) / (2.0 * F)
            dln_fk_dK = (1.0 - b) / (2.0 * K)
            dln_fk_db = -0.5 * np.log(F * K)

            # d(ln sigma)/d(param) = [1/a for alpha] + dz/z - dx/x - d(ln fk) - dw1/w1 + T*dw2/(1+w2T)
            greeks[SabrMetrics.D_LN_SIGMA_D_ALPHA] = (
                1.0 / a
                + risk_z[SabrMetrics.DALPHA] / z
                - risk_w1[SabrMetrics.DALPHA] / w1
                - risk_x[SabrMetrics.DALPHA] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DALPHA]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_BETA] = (
                risk_z[SabrMetrics.DBETA] / z
                - dln_fk_db
                - risk_w1[SabrMetrics.DBETA] / w1
                - risk_x[SabrMetrics.DBETA] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DBETA]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_RHO] = (
                risk_z[SabrMetrics.DRHO] / z
                - risk_w1[SabrMetrics.DRHO] / w1
                - risk_x[SabrMetrics.DRHO] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DRHO]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_NU] = (
                risk_z[SabrMetrics.DNU] / z
                - risk_w1[SabrMetrics.DNU] / w1
                - risk_x[SabrMetrics.DNU] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DNU]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_FORWARD] = (
                risk_z[SabrMetrics.DFORWARD] / z
                - dln_fk_dF
                - risk_w1[SabrMetrics.DFORWARD] / w1
                - risk_x[SabrMetrics.DFORWARD] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DFORWARD]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKE] = (
                risk_z[SabrMetrics.DSTRIKE] / z
                - dln_fk_dK
                - risk_w1[SabrMetrics.DSTRIKE] / w1
                - risk_x[SabrMetrics.DSTRIKE] / x_val
                + factor_w2 * risk_w2[SabrMetrics.DSTRIKE]
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = w2 / (1.0 + w2 * T)

            # d²(ln sigma)/dK²:
            # d²(ln(z/x))/dK² + (1-b)/(2K²) - d²(ln w1)/dK² + T*d²w2/dK²/(1+w2T) - T²*(dw2/dK)²/(1+w2T)²
            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = (
                (risk_z[SabrMetrics.DSTRIKESTRIKE] * z - risk_z[SabrMetrics.DSTRIKE] ** 2) / z ** 2
                - (risk_x[SabrMetrics.DSTRIKESTRIKE] * x_val - risk_x[SabrMetrics.DSTRIKE] ** 2) / x_val ** 2
                + (1.0 - b) / (2.0 * K ** 2)
                - (risk_w1[SabrMetrics.DSTRIKESTRIKE] * w1 - risk_w1[SabrMetrics.DSTRIKE] ** 2) / w1 ** 2
                + factor_w2 * risk_w2[SabrMetrics.DSTRIKESTRIKE]
                - factor_w2 ** 2 * risk_w2[SabrMetrics.DSTRIKE] ** 2
            )

        return sigma, greeks