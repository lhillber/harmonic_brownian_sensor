#!/usr/bin/env python
# coding: utf-8
# Compute measures on bownian motion time series data.
# By Logan Hillberry, 3 November 2020

import os
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import curve_fit, minimize
from scipy.signal import decimate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.mlab import psd, stride_windows
from functools import wraps
from copy import copy
from numba import jit
from scipy.signal import welch

welch_psd = psd

# constants
PI = np.pi
kB = 1.382e-23
RHO = 2000.0  # Bangs Labs value

# dynamic viscosity vs temperature
mair = 28.9647 * 1e-3 / 6.02214076e23
sig = 3.617e-10
eps_per_kB = 97.0

def get_eta2(T):
    Ts = T / eps_per_kB
    omg = 1.16145 * Ts**(-0.14874) + 0.52487 * np.exp(-0.77320*Ts) + 2.16178*np.exp(-2.43787*Ts)
    return 5 * np.sqrt(kB * T * mair/PI) / (16* sig**2*omg)


def get_eta(T, etap=1.83245e-5,Tp=23+273.15, S=110.4):
    return etap*(T/Tp)**(3/2) * (Tp + S) / (T+S)


# functions for fitting
def gaussian_func(x, var):
    return np.exp(-x * x / (2 * var)) / (np.sqrt(2 * PI * var))


def psd_func(f, A, kappa, mass, gamma, T):
    return (
        4
        * kB
        * T
        * gamma
        / (
            A
            * A
            * ((mass * (2 * PI * f) ** 2 - kappa) ** 2 + (2 * PI * f * gamma) ** 2)
        )
    )


def psd_noise_func(f, A, kappa, mass, B, gamma, T):
    return B + 4 * kB * T * gamma / (
        A * A * ((mass * (2 * PI * f) ** 2 - kappa) ** 2 + (2 * PI * f * gamma) ** 2)
    )


def psd_abc_func(f, a, b, c, **kwargs):
    return 1 / np.abs((a + b * f ** 2 + c * f ** 4))


def psd_radius_func(f, kappa, R, A, T):
    eta = get_eta(T)
    return 4 * kB * T * (6 * PI * eta * R) / (
        1/A/A
        * (
            ((4 / 3 * PI * RHO * R ** 3) * (2 * PI * f) ** 2 - kappa) ** 2
            + (2 * PI * f * (6 * PI * eta * R)) ** 2
        )
    )


def psd_density_func(f, kappa, rho, A, R, T):
    eta = get_eta(T)
    return 4 * kB * T * (6 * PI * eta * R) / (
        1/A/A
        * (
            ((4 / 3 * PI * rho * R ** 3) * (2 * PI * f) ** 2 - kappa) ** 2
            + (2 * PI * f * (6 * PI * eta * R)) ** 2
        )
    )


def lorentzian_radius_func(f, kappa, R, A, T, **kwargs):
    eta = get_eta(T)
    return (
        4
        * kB
        * T
        * (6 * PI * eta * R)
        / (A * A * (((6 * PI * eta * R) * 2 * PI * f) ** 2 + kappa ** 2))
    )


def lorentzian_func(f, A, kappa, gamma, B, T):
    return B, 4 * kB * T * gamma / (A * A * ((gamma * 2 * PI * f) ** 2 + kappa ** 2))


def pacf_func(t, Omega, Gamma):
    omega1 = np.sqrt(Omega ** 2 - Gamma ** 2 / 4.0)
    return np.exp(-0.5 * Gamma * t) * (
        np.cos(omega1 * t) + 0.5 * Gamma * np.sin(omega1 * t) / omega1
    )


def pacf_od_func(t, Omega, Gamma):
    omega1 = np.sqrt(Gamma ** 2 / 4.0 - Omega ** 2)
    tauplus = 2.0 / (Gamma * (1 + 2.0 * omega1 / Gamma))
    tauminus = 2.0 / (Gamma * (1 - 2.0 * omega1 / Gamma))
    return (np.exp(-t / tauminus) / tauplus - np.exp(-t / tauplus) / tauminus) / (
        2 * omega1
    )


def vacf_func(t, Omega, Gamma):
    omega1 = np.sqrt(Omega ** 2 - Gamma ** 2 / 4)
    return np.exp(-0.5 * Gamma * t) * (
        np.cos(omega1 * t) - 0.5 * Gamma * np.sin(omega1 * t) / omega1
    )


def vacf_od_func(t, Omega, Gamma):
    omega1 = np.sqrt(Gamma ** 2 / 4.0 - Omega ** 2)
    tauplus = 2.0 / (Gamma * (1 + 2.0 * omega1 / Gamma))
    tauminus = 2.0 / (Gamma * (1 - 2.0 * omega1 / Gamma))
    return -np.exp(-t / tauminus) / (2 * omega1 * tauminus) + np.exp(-t / tauplus) / (
        2 * omega1 * tauplus
    )


def msd_func(t, A, Omega, Gamma):
    # A = 2 kB T / M Omega^2 in theory
    omega1 = np.sqrt(Omega ** 2 - Gamma ** 2 / 4)
    return A * (
        1
        - np.exp(-0.5 * Gamma * t)
        * (np.cos(omega1 * t) + 0.5 * Gamma * np.sin(omega1 * t) / omega1)
    )


def msd_od_func(t, A, Omega, Gamma):
    omega1 = np.sqrt(Gamma ** 2 / 4.0 - Omega ** 2)
    tauplus = 2.0 / (Gamma * (1 + 2.0 * omega1 / Gamma))
    tauminus = 2.0 / (Gamma * (1 - 2.0 * omega1 / Gamma))
    return A * (
        1
        - np.exp(-t / tauminus) / (2 * omega1 * tauplus)
        + np.exp(-t / tauplus) / (2 * omega1 * tauminus)
    )


fitting_funcs = {
    "psd": psd_func,
    "psd_radius": psd_radius_func,
    "psd_density": psd_density_func,
    "psd_abc": psd_abc_func,
    "psd_lorentzian": lorentzian_func,
    "psd_lorentzian_radius": lorentzian_radius_func,
    "hist": gaussian_func,
    "hist_x": gaussian_func,
    "hist_v": gaussian_func,
    "pacf": pacf_func,
    "vacf": vacf_func,
    "msd": msd_func,
    "msd_od": msd_od_func,
    "pacf_od": pacf_od_func,
    "vacf_od": vacf_od_func,
}




@jit(nopython=True)
def MSD_jit(x):
    N = x.size
    lags = np.arange(1, N)
    msd = np.zeros(N - 1, dtype=np.float32)
    for lag in lags:
        msd_el = 0.0
        for i in range(x.size - lag):
            msd_el += (x[i + lag] - x[i]) ** 2
        msd[lag - 1] = msd_el / (N - lag)
    return msd


MSD_jit(np.array([1.0, 2.9, 3.0, 4.0]))


@jit(nopython=True)
def ACF_jit(x, r):
    N = x.size
    v = np.var(x)
    m = np.mean(x)
    lags = np.arange(1, N)
    acf = np.zeros(N - 1, dtype=np.float32)
    for lag in lags:
        acf_el = 0.0
        for i in range(x.size - lag):
            acf_el += (x[i] - m) * (x[i + lag] - m)
        acf[lag - 1] = acf_el / (N - lag) / v
    return acf


ACF_jit(np.array([1.0, 2.9, 3.0, 4.0]), 1.0)


def parse_fname(fname):
    """
    Returns dictionary of parameters encoded in file name.
    Assumes structure:
    der/<name1>-<unit1><val1>_<name2>-<unit2><val2>_<note1>_<note2>/trial_v<version>.txt
    No numbers are allowed in the name, unit, or note fields
    """
    #assumes <path tail>/data/<date>/<path head>/<fname>.dat
    ders, fname = os.path.split(fname)
    parts = []
    while ders != os.path.sep:
        ders, der = os.path.split(ders)
        parts.append(der)
    parts = parts[::-1]
    date = parts[parts.index("data")+1]
    notes = parts[parts.index("data")+2:]
    params = {}
    name = fname.split(".dat")[0]
    for puv in name.split("_"):
        for i, c in enumerate(puv):
            if c.isdigit():
                break
        if i == len(puv) - 1:
            notes.append(puv)
        else:
            params[puv[:i]] = float(puv[i:])
    params["notes"] = notes
    params["date"] = date
    return params

def psdfunc(f, k, rho, A, T, R):
    m = 4*np.pi*rho*R**3/3
    eta = get_eta(T)
    gamma = 6*np.pi*eta*R
    omega = 2*np.pi * f
    denom = (m*omega**2 - k)**2 + (gamma*omega)**2
    return 4 * kB * T * gamma / (A*A*denom)

class TimeSeries:
    def __init__(self, fname=None, rate=None, norm=None, verbose=False):
        self.verbose = verbose
        if fname is None:
            if self.verbose:
                print("Empty TimeSeries")
            self.x = np.array([])
            self.t = np.array([])
            self.params = {}
        else:
            self.load(fname, rate=rate, norm=norm)

    def r(self, key):
        t, _ = self.get_tx(key)
        return 1.0 / (t[1] - t[0])


    @property
    def T(self):
        try:
            T = self.params["Tair-C"] + 273.15
        except KeyError:
            return self._T
        return T

    @T.setter
    def T(self, T):
        self._T = T

    @property
    def R(self):
        try:
            R = self.params["D-um"] * 1e-6 / 2.0
        except KeyError:
            R = self._R
        return R

    @property
    def eta(self):
        return get_eta(self.T)

    @R.setter
    def R(self, R):
        self._R = R

    def get_k(self, a, b, c):
        return 12 * np.pi**2 * self.eta * self.R * np.sqrt(a/self.get_d2(a,b,c))

    def get_rho(self, a, b, c):
        return 9 * self.eta * np.sqrt(c/self.get_d2(a,b,c)) / (4*np.pi*self.R**2)

    def get_A(self, a, b, c):
        Ainv2 = (6 * np.pi**3 * self.eta * self.R) / (kB * self.T * self.get_d2(a, b, c))
        return 1 / np.sqrt(Ainv2)

    def get_d2(self, a, b, c):
        return b + 2 *np.sqrt(a*c)


    def calibrate(self, mask=None):
        abcx = self.get_abc("x", mask=mask)
        kx0 = self.get_k(*abcx)
        rhox0 = self.get_rho(*abcx)
        Ax0 = self.get_A(*abcx)
        abcy = self.get_abc("y", mask=mask)
        ky0 = self.get_k(*abcy)
        rhoy0 = self.get_rho(*abcy)
        Ay0 = self.get_A(*abcy)
        rho0 = (rhox0 + rhoy0)/2
        p0 = [kx0, ky0, rho0, Ax0, Ay0]
        def model(f, kx, ky, rho, Ax, Ay):
            return np.array(
                [psdfunc(f, kx, rho, Ax, self.T, self.R),
                 psdfunc(f, ky, rho, Ay, self.T, self.R)]
                 ).ravel()
        vals = np.array([self.x_psd[mask], self.y_psd[mask]]).ravel()
        indep = self.x_freq[mask]
        popt, pcov = curve_fit(model, indep, vals, p0=p0,
            bounds=((0,0,0,0,0), (np.inf, np.inf, np.inf,np.inf,np.inf)))
        self.kx = popt[0]
        self.ky = popt[1]
        self.rho = popt[2]
        self.Ax = popt[3]
        self.Ay = popt[4]
        return popt, pcov


    @property
    def mass(self):
        return self.rho * 4 * PI * self.R ** 3 / 3

    @property
    def gamma(self):
        return 6 * PI * self.eta * self.R

    @property
    def taup(self):
        return self.mass / self.gamma

    @property
    def Gamma(self):
        return 1.0 / self.taup

    def reset(self):
        self.x = self.xbak[:]
        self.y = self.ybak[:]
        self.tx = self.txbak[:]
        self.ty = self.tybak[:]

    def resize(self, Npts=2 ** 24):
        d = Npts - self.size
        if d > 0:
            x = np.zeros(Npts, dtype=np.float32)
            t = np.zeros(Npts, dtype=np.float32)
            x[: self.size] = self.x
            t[: self.size] = self.t
            for i in range(1, d + 1):
                x[-i] = self.x[-1]
                t[-i] = self.t[-1] + (1 + d - i) / self.r
            self.x = x
            self.t = t
            self.xbak = x
            self.tbak = t

    def load(self, fname, rate=None, norm=None):
        if self.verbose:
            print("Loading")
            print(fname)
        params = parse_fname(fname)
        self.params = params
        if rate is None:
            rate = self.params["r-Sps"]

        X = np.fromfile(fname, dtype=">d")
        x = X[:X.size//2]
        y = X[X.size//2:]
        x -= np.mean(x)
        y -= np.mean(y)
        tx = np.arange(0.0, x.size) / rate
        ty = np.arange(0.0, y.size) / rate
        self.txbak = copy(tx)
        self.tybak = copy(ty)
        self.xbak = copy(x)
        self.ybak = copy(y)
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.tx = np.array(tx, dtype=np.float32)
        self.ty = np.array(ty, dtype=np.float32)
        return tx, x, ty, y

    def PSD(self, coord="x", tmax=None, detrend="linear", window="hamming"):
        # split data into partitions of length tmax
        t = getattr(self, self.get_tkey(coord))
        r = self.r(coord)
        if tmax is None:
            tmax = t[-1]
        Npts = min(t.size, int(tmax * r))
        if self.verbose:
            print("Calculating PSD...")
        tpart, xpart = self.partition(coord, Npts=Npts)
        Navg = len(xpart)
        Npts = len(xpart[0])
        if self.verbose:
            print(f" averaging {Navg} segments of length {Npts}")
        tp = np.array(tpart[0])
        freq = fft.rfftfreq(tp.size, 1 / r)
        psdavg = 0
        # average PSD of each partions
        for xp in xpart:
            # optional detrend for each segment
            if detrend is None:
                x = xp
            if detrend == "const":
                x = xp - xp.mean()
            if detrend == "linear":
                m, b = np.polyfit(tp, xp, deg=1)
                x = xp - (m * tp + b)
                # weltch detrend version:
                #ll = np.arange(xp.size, dtype=float)
                #C = np.cov(ll, xp, bias=1)
                #b = C[0, 1]/C[0, 0]
                #a = xp.mean() - b*ll.mean()
                #x = xp - (b*ll + a)

            # Hamming window
            if window == "hamming":
                w = np.array([0.54 - 0.46*np.cos(2*PI*j/(len(x) - 1)) for j in range(len(x))])
            # rectangular window
            elif window in (None, "none", "rect"):
                w = np.ones(x.shape)
            # real FT
            xk = fft.rfft(w*x, x.size)
            # rescale: numer. for 1-sided, denom. for window correction
            scale = 2 / (r * (np.abs(w)**2).sum())
            psd = scale * (xk*np.conj(xk)).real
            # other built in algs:
            #freq, psd0 = welch(xp, fs=r, nperseg=x.size, detrend="linear", noverlap=0)
            #psd1, freq = welch_psd(xp, Fs=r, NFFT=x.size, detrend="linear")
            psdavg += psd / Navg
        # clip zero freq bin = average of signal
        psdavg = psdavg[1:]
        freq = freq[1:]
        setattr(self, f"{coord}_freq", freq)
        setattr(self, f"{coord}_psd", psdavg)
        setattr(self, f"{coord}_Npsd", Navg)
        return freq, psdavg

    # def AVAR(self, x=None, r=None):
    #    print("WARNING: use ADVEV")
    #    if self.verbose:
    #        print("Calculating AVAR...")
    #    if x is None:
    #        x = self.x
    #    if r is None:
    #        r = self.r
    #    X = np.cumsum(x)
    #    Nparts = min(1000, x.size - 2)
    #    parts = np.logspace(
    #        np.log10(Nparts), np.log10((x.size - Nparts) // 10), Nparts, base=10
    #    )
    #    parts = np.unique(np.round(parts)).astype(np.int64)
    #    Nparts = parts.size
    #    avar = np.zeros(parts.size)
    #    for i, p in enumerate(parts):
    #        v = X[2 * p :] - 2 * X[p:-p] + X[: -2 * p]
    #        avar[i] = np.mean(v ** 2) / (p * p)
    #    avar *= r * r / 2.0
    #    tavar = parts / r
    #    self.tavar = tavar
    #    self.avar = avar
    #    return tavar, avar

    def AVAR(self, key, fit=False, Nbins=20):
        if self.verbose:
            print("Calculating ADEV...")
        if fit:

            def var_func(x):
                hist, edges = np.histogram(x, bins=Nbins, density=True)
                bins = edges[:-1] + np.diff(edges) / 2
                popt, pcov = curve_fit(gaussian_func, bins, hist, p0=[np.var(x)])
                return popt[0]

        else:
            var_func = np.var

        Npts_list = np.linspace(self.size // 2**14, self.size // 4, 400)
        Npts_list = np.unique(np.round(Npts_list)).astype(np.int64)
        Npts_list = Npts_list[Npts_list >= 10]
        tavar = Npts_list / self.r(key)
        ps = np.zeros_like(tavar)
        dps = np.zeros_like(tavar)
        pavar = np.zeros_like(tavar)
        for j, Npts in enumerate(Npts_list):
            tparts, xparts = self.partition(key, Npts=Npts)
            pvals = np.array([var_func(x) for x in xparts])
            dps[j] = np.std(pvals)
            ps[j] = np.mean(pvals)
            pavar[j] = np.mean(0.5 * (pvals[1:] - pvals[:-1]) ** 2)
        setattr(self, f"t{key}avar", tavar)
        setattr(self, f"{key}s", ps)
        setattr(self, f"d{key}s", dps)
        setattr(self, f"{key}avar", pavar)
        return tavar, pavar, dps, ps

    def MSD(self, key="x", r=None, taumax=1e-3):
        tmax = taumax / 0.05
        Npts = int(taumax * self.r(key))
        if self.verbose:
            print("Calculating MSD...")
            if tmax > self.t[-1]:
                print("Poor statistics for late times")
        if r is None:
            r = self.r(key)
        tpart, xpart = self.partition(key, Npts=Npts, tmax=tmax)
        Navg = len(xpart)
        Npts = len(xpart[0])
        msdavg = 0
        if self.verbose:
            print(f" averaging {Navg} segments of length {Npts}")
        for xp in xpart:
            msdavg += MSD_jit(xp) / Navg
        tmsd = np.arange(1, len(msdavg) + 1) / r
        self.msd = msdavg
        self.tmsd = tmsd
        return tmsd, msdavg

    def ACF2(self, x=None, t=None, key=None, normalize=True, tmax=1e-3):
        partkey = key
        if x is None:
            if key in ("V", "dxdt", "v"):
                key = "v"
                x = self.v
                if t is None:
                    t = self.tv
            elif key in ("P", "x", "p"):
                key = "p"
                partkey = "x"
                x = self.x
                if t is None:
                    t = self.t
            else:
                if self.verbose:
                    print("Calculating ACF...")
        Npts = min(x.size, int(tmax / (t[1] - t[0])))
        tacf, xpart = self.partition(partkey, Npts=Npts)
        tacf = tacf[0]
        Navg = len(xpart)
        Npts = len(t)
        acfavg = 0
        if self.verbose:
            print(f" averaging {Navg} segments of length {Npts}")
        for xp in xpart:
            denom = xp.size * np.ones(xp.size) - np.arange(0, xp.size)
            acf = np.correlate(xp, xp, mode="full")
            acf = acf[acf.size // 2 :] / denom
            acfavg += acf
        acfavg /= Navg
        if normalize:
            acfavg /= np.var(x)
        setattr(self, key + "acf", acfavg)
        setattr(self, "t" + key + "acf", tacf)
        return tacf, acfavg

    def ACF(self, x=None, r=None, key=None, normalize=True, tmax=1e-3):
        if r is None:
            r = self.r(key)
        partkey = key
        if x is None:
            if key in ("V", "dxdt", "v"):
                key = "v"
                x = self.v
            elif key in ("P", "x", "p"):
                key = "p"
                partkey = "x"
                x = self.x
            else:
                if self.verbose:
                    print("Calculating ACF...")
        tacf, xpart = self.partition(partkey, tmax=tmax)
        tacf = tacf[0]
        Navg = len(xpart)
        Npts = len(tacf)
        acfavg = 0
        if self.verbose:
            print(f" averaging {Navg} segments of length {Npts}")
        for xp in xpart:
            acfavg += ACF_jit(x, r)
        acfavg /= Navg
        setattr(self, key + "acf", acfavg)
        setattr(self, "t" + key + "acf", tacf)
        return tacf, acfavg

    def PACF(self, tmax=1e-3):
        if self.verbose:
            print("Calculating PACF...")
        return self.ACF(key="x", tmax=tmax)

    def VACF(self, tmax=1e-3):
        if self.verbose:
            print("Calculating VACF...")
        return self.ACF(key="v", tmax=tmax)

    def HIST(self, key, Nbins=100, tmin=None, tmax=None, density=True, weights=None):
        t, x = self.get_tx(key, tmin=tmin, tmax=tmax)
        hist, edges = np.histogram(x, bins=Nbins, density=density, weights=weights)
        bins = edges[:-1] + np.diff(edges) / 2
        if not density:
            hist = hist / np.sum(hist)
        setattr(self, "hist_" + key, hist)
        setattr(self, "bins_" + key, bins)
        return bins, hist

    def firstdiff(self, key="x", dt=None, acc=8):
        if self.verbose:
            print("Calculating first derivative...")
        t, x = self.get_tx(key)
        if dt is None:
            dt = 1.0 / self.r(key)
        assert acc in [2, 4, 6, 8]
        coeffs = [
            [1.0 / 2],
            [2.0 / 3, -1.0 / 12],
            [3.0 / 4, -3.0 / 20, 1.0 / 60],
            [4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
        ]
        v = np.sum(
            np.array(
                [
                    (
                        coeffs[acc // 2 - 1][k - 1] * x[k * 2 :]
                        - coeffs[acc // 2 - 1][k - 1] * x[: -k * 2]
                    )[acc // 2 - k : len(x) - (acc // 2 + k)]
                    / dt
                    for k in range(1, acc // 2 + 1)
                ]
            ),
            axis=0,
        )
        tv = t[acc // 2 : -acc // 2]
        setattr(self, "v"+key, v)
        setattr(self, "t"+"v"+key, tv)
        return tv, v

    def WKcheck(self, tol=1e-4):
        varsignal = np.var(self.x)
        psd_integral = simps(self.psd, x=self.freq)
        absdiff = abs(varsignal - psd_integral)
        check = absdiff < tol
        print(f"check: {check} for tol {tol}")
        print(f"variance of signal [V]: {varsignal}")
        print(f"integral of PSD [V]: {psd_integral}")

    def moving_average(self, x=None, t=None, Npts=100, inplace=False):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        x2 = np.convolve(self.x, np.ones((Npts,)) / Npts, mode="valid")
        t2 = t[: x2.size]
        if inplace:
            self.x = x2
            self.t = t2
        return t2, x2

    def stride(
        self,
        key="x",
        Nstrides=128,
        Noverlap=0,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
    ):
        t, x = self.get_tx(key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        ts = stride_windows(t, n=Nstrides, noverlap=Noverlap, axis=0)
        xs = stride_windows(x, n=Nstrides, noverlap=Noverlap, axis=0)
        return ts, xs

    def partition(
        self, key="x", Npts=128, Noverlap=0, tmin=None, tmax=None, fmin=None, fmax=None
    ):
        ts, xs = self.stride(
            key=key,
            Nstrides=Npts,
            Noverlap=Noverlap,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
        )
        return np.transpose(ts), np.transpose(xs)

    def bin_average(self, key="x", Npts=128, inplace=False):
        tkey = self.get_tkey(key)
        if Npts in (1, None):
            t2, x2 = self.get_tx(key)
        else:
            t2, x2 = self.partition(key, Npts=Npts)
            x2 = np.mean(x2, axis=1)
            t2 = np.mean(t2, axis=1)
            setattr(self, key, x2)
            setattr(self, tkey, t2)
        return t2, x2

    def bin_std(self, key="x", Npts=128, inplace=False):
        if Npts in (1, None):
            inplace = False
            return None
        else:
            t2, x2 = self.partition(key, Npts=Npts)
            x2 = np.std(x2, axis=1)
        if inplace:
            setattr(self, "d" + key, x2)
        return x2

    def logbin_average(self, key="x", Npts=100, inplace=False):
        if type(key) == str:
            t2, x2 = self.get_tx(key)
        else:
            x2 = key
        ndecades = np.log10(x2.size) - np.log10(1)
        npoints = int(ndecades) * Npts
        parts = np.logspace(
            np.log10(1), np.log10(x2.size), num=npoints, endpoint=True, base=10
        )
        # parts = np.logspace(
        #    np.log10(Npts), np.log10((x2.size - Npts) // 10), Npts, base=10
        # )
        parts = np.unique(np.round(parts)).astype(np.int64)
        x2 = np.array(
            [np.mean(x2[parts[i] : parts[i + 1]]) for i in range(len(parts) - 1)]
        )

        if type(key) == str:
            t2 = np.array(
                [np.mean(t2[parts[i] : parts[i + 1]]) for i in range(len(parts) - 1)]
            )
            if inplace:
                tkey = self.get_tkey(key)
                setattr(self, key, x2)
                setattr(self, tkey, t2)
            return t2, x2
        return x2

    def logbin_std(self, key="x", Npts=100, inplace=False):
        t2, x2 = self.get_tx(key)

        # parts = np.logspace(
        #    np.log10(Npts), np.log10((x2.size - Npts) // 10), Npts, base=10
        # )

        ndecades = np.log10(x2.size) - np.log10(1)
        npoints = int(ndecades) * Npts
        parts = np.logspace(
            np.log10(1), np.log10(x2.size), num=npoints, endpoint=True, base=10
        )

        parts = np.unique(np.round(parts)).astype(np.int64)
        x2 = np.array(
            [np.std(x2[parts[i] : parts[i + 1]]) for i in range(len(parts) - 1)]
        )

        if inplace:
            setattr(self, "d" + key, x2)
        return x2

    def get_tkey(self, key):
        coord, *meas = key.split("_")
        if len(meas) == 1:
            meas = meas[0]
        elif len(meas) == 0:
            meas = coord
            tkey = f"t{coord}"
        if meas[0] == "d":
            meas = meas[1:]
        if meas == "psd":
            tkey = f"{coord}_freq"
        elif key == "hist":
            tkey = f"{coord}_bins"
        else:
            tkey = "t" + key
        return tkey

    def get_tx(self, key, tmin=None, tmax=None, fmin=None, fmax=None):
        tkey = self.get_tkey(key)
        x = getattr(self, key)
        t = getattr(self, tkey)
        if tmin is None:
            tmin = t[0]
        if tmax is None:
            tmax = t[-1]
        if fmin is not None:
            tmin = fmin
        if fmax is not None:
            tmax = fmax
        mask = np.logical_and(t >= tmin, t <= tmax)
        return t[mask], x[mask]

    def S(self, p, q, freq, psd):
        return np.mean(freq ** (2 * p) * psd ** q)

    def getS(self, freq, psd):
        S02 = self.S(0, 2, freq, psd)
        S12 = self.S(1, 2, freq, psd)
        S22 = self.S(2, 2, freq, psd)
        S32 = self.S(3, 2, freq, psd)
        S42 = self.S(4, 2, freq, psd)
        S01 = self.S(0, 1, freq, psd)
        S11 = self.S(1, 1, freq, psd)
        S21 = self.S(2, 1, freq, psd)
        Smat = np.array([[S02, S12, S22], [S12, S22, S32], [S22, S32, S42]])
        Denom = (
            S02 * S22 * S42
            - S02 * S32 ** 2
            - S12 ** 2 * S42
            + 2 * S12 * S22 * S32
            - S22 ** 3
        )
        C0 = S22 * S42 - S32 ** 2
        C1 = S22 * S32 - S12 * S42
        C2 = S12 * S32 - S22 ** 2
        C3 = S12 * S22 - S02 * S32
        C4 = S02 * S22 - S12 ** 2
        C5 = S02 * S42 - S22 ** 2
        C = np.array([[C0, C1, C2], [C1, C5, C3], [C2, C3, C4]])
        Sinvmat = C / Denom
        Svec = np.array([S01, S11, S21])
        return Smat, Sinvmat, Svec

    def get_abc(self, coord, fmin=0, fmax=5e7, cov=False, optimize=False, mask=None):
        freq, psd = self.get_tx(f"{coord}_psd", fmin=fmin, fmax=fmax)
        if mask is not None:
            freq = freq[mask]
            psd = psd[mask]
        Npsd = getattr(self, f"{coord}_Npsd")
        Smat, Sinvmat, Svec = self.getS(freq, psd)
        Sinvmat *= (Npsd + 1) / Npsd
        abc = Sinvmat.dot(Svec)
        if optimize:
            def obj(P):
                return np.sum(psd/psd_abc_func(freq, *P) + np.log(psd_abc_func(psd, *P)))
            res = minimize(obj, abc, bounds=((0, np.inf),(-np.inf,np.inf),(0, np.inf)))
            abc = res.x
        _, pSinvmat, __ = self.getS(freq, psd_abc_func(freq, *abc))
        abc_cov = (Npsd + 3) * pSinvmat / (Npsd + 1) / freq.size
        if cov:
            return abc, abc_cov
        else:
            return abc

    def fit(
        self,
        key,
        p0,
        bounds=None,
        weighted=False,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        fixed_kwargs=dict(),
        ML=False,
    ):
        t, x = self.get_tx(key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)

        def func(t, *args):
            return fitting_funcs[key](t, *args, **fixed_kwargs)


        if ML == True:
            assert key[:3] == "psd"
            Nfiles = self.Nfiles
            Smat, Sinvmat, Svec = self.getS(t, x)
            Sinvmat *= (Nfiles + 1) / Nfiles
            P0 = Sinvmat.dot(Svec)
            # def obj(P):
            #    return np.sum(x/func(t,*P) + np.log(func(t, *P)))
            # res = minimize(obj, P0, bounds=((0, np.inf),(-np.inf,np.inf),(0, np.inf)))
            # popt = res.x
            popt = P0
            _, pSinvmat, __ = self.getS(t, func(t, *popt))
            pcov = (Nfiles + 3) * pSinvmat / (Nfiles + 1) / t.size / 2

        else:  # least sqs
            if weighted:
                try:
                    _, dx = self.get_tx(
                        "d" + key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax
                    )
                except AttributeError:
                    if self.verbose:
                        print("no dx found")
                    dx = None
            else:
                dx = None
            popt, pcov = curve_fit(
                func, t, x, p0=p0, sigma=dx, absolute_sigma=True, bounds=bounds
            )

        setattr(self, "popt_" + key, popt)
        setattr(self, "pcov_" + key, pcov)
        setattr(self, "fixed_kwargs_" + key, fixed_kwargs)
        return popt, pcov

    def model(self, key, t):
        try:
            popt = getattr(self, "popt_" + key)
            fixed_kwargs = getattr(self, "fixed_kwargs_" + key)
        except KeyError:
            print("no model available!")
            return
        return fitting_funcs[key](t, *popt, **fixed_kwargs)

    def plot_fit(
        self,
        key,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        ax=None,
        fig=None,
        figsize=None,
        popt=None,
        scale="lin",
        fixed_kwargs=dict(),
        **kwargs,
    ):
        if figsize is None:
            figsize = (1.618 * 3, 3)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        if fig is None:
            fig = plt.gcf()
        if popt is None:
            popt = getattr(self, "popt_" + key)
        t, x = self.get_tx(key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        if scale in ("log", "logx"):
            ax.set_yscale("log")
            ts = np.linspace(t[0], t[-1], 300)
        elif scale in ("logt",):
            if t[0] > 0:
                t0 = t[0]
            else:
                t0 = t[1]
            ax.set_xscale("log")
            ts = np.geomspace(t0, t[-1], 300)
        elif scale in ("loglog", "logxt"):
            if t[0] > 0:
                t0 = t[0]
            else:
                t0 = t[1]
            ts = np.geomspace(t0, t[-1], 300)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ts = np.linspace(t[0], t[-1], 300)

        def func(t, *args):
            return fitting_funcs[key](t, *args, **fixed_kwargs)

        ax.plot(ts, func(ts, *popt), **kwargs)
        return fig, ax

    def plot(
        self,
        key,
        errorbar=False,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        ax=None,
        fig=None,
        Npts_log=None,
        figsize=None,
        scale="lin",
        marker="o",
        ms=2,
        ls="none",
        **kwargs,
    ):
        if figsize is None:
            figsize = (1.618 * 3, 3)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        if fig is None:
            fig = plt.gcf()
        t, x = self.get_tx(key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        if Npts_log is not None:
            t, x = self.logbin_average(key, Npts=Npts_log)
        if key.split("_")[0] == "hist":
            ax.step(t, x, marker=marker, ms=ms, where="mid", **kwargs)
        else:
            if errorbar:
                try:
                    _, dx = self.get_tx(
                        "d" + key, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax
                    )
                except AttributeError:
                    dx = np.zeros_like(x)
                ax.errorbar(t, x, yerr=dx, marker=marker, ms=ms, ls=ls, **kwargs)
            else:
                ax.plot(t, x, marker=marker, ms=ms, ls=ls, **kwargs)
        if scale in ("log", "logx"):
            ax.set_yscale("log")
        elif scale in ("logt", "logf"):
            ax.set_xscale("log")
        elif scale in ("loglog", "logxt"):
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(key)
        return fig, ax


def swap(D):
    D2 = TimeSeries()
    D2.__dict__ = D.__dict__
    D = D2
    del D2
    return D


def find_files(der):
    """Return the full path to all files in directory der."""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(der):
        for file in f:
            files.append(os.path.join(r, file))
    return files


def find_ders(inder):
    """Return the full path to all directories in directory der."""
    ders = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(inder):
        for der in d:
            if len(r.split(os.path.sep)) == len(inder.split(os.path.sep)):
                ders.append(os.path.join(r, der))

    return ders


def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(
                pp, format="pdf", bbox_inches="tight", bbox_extra_artist=extra_artist
            )
        else:
            fig.savefig(pp, format="pdf", bbox_extra_artist=extra_artist)
        if clf == True:
            fig.clf()

    pp.close()
