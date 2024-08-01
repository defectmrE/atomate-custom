import numpy as np
import scipy.constants as const
import itertools

from pymatgen.electronic_structure.core import Spin


def vaspwfc_analysis(ref_ind, kpoints, spin, i_band, f_band, wav_file, lgamma):
    """
    compute transition dipole moment and lifetime
    Args:
        ref_ind: refractive index
        kpoints: kpoint index (int)
        spin: spin index, 0 for down, 1 for up (int)
        i_band: initial band index (int), starts from 1
        f_band: final band index (int), starts from 1
        wave_file: WAVECAR file name (str)
        lgamma: If it is calcuated using vasp_gam (boolean)


    Returns:

    """
    from vaspwfc import vaspwfc

    epsilon0 = const.epsilon_0  # vacuum permittivity
    h = const.Planck
    c = const.speed_of_light
    eV_to_Hz = const.physical_constants['electron volt-hertz relationship'][0]

    wav = vaspwfc(wav_file, lgamma=lgamma)
    res = wav.TransitionDipoleMoment([spin, kpoints, i_band],
                                     [spin, kpoints, f_band],
                                     norm=True)
    mu = np.linalg.norm(res[-1])
    musq = mu ** 2

    transition_e = res[2]
    freq = transition_e * eV_to_Hz
    A = (2 * ref_ind * (2 * const.pi) ** 3 * freq ** 3 * musq * (3.33e-30) ** 2) / (
            3 * epsilon0 * h * c ** 3)  # The 3.33e-30 converts to Debye
    tau = 1 / A * 1e9  # in unit of ns

    return transition_e, mu, tau


def get_all_tdm_and_lifetime(ref_ind, kpoints, wav_file, lgamma, int_eigenval, vbm, cbm, emin=-1, emax=1,
                             truncated_bands=10):
    """

    Args:
        ref_ind: refractive index of the materials (float)
        kpoints: kpoints index, gamma point is 1 (int)
        wav_file: WAVECAR file name (str)
        lgamma: If it is calcuated using vasp_gam (boolean)
        int_eigenval: integerized eigenvalue files
        vbm: valence band maximum in eV
        cbm: conduction band minimum in eV
        emin: lowerbound energy below VBM
        emax: upperbound energy above CBM
        truncated_bands: limit occupied states and unoccupied states, default to 10

    Returns:
        dictionary contains exitation order, initial and final state band index, mu and tau.
    """

    tdm_tau_dict = {}

    for spin, egv in int_eigenval.items():
        homo_band_ind = np.where(np.diff(egv[:, 1]) == -1)[0][0]
        band_range = np.where((vbm + emin < egv[:, 0]) & (egv[:, 0] < cbm + emax))[0]

        occu_bands = []
        empty_bands = []

        for band_ind in band_range:
            if band_ind <= homo_band_ind:
                occu_bands.append(band_ind)
            else:
                empty_bands.append(band_ind)

        occu_bands = occu_bands[::-1][:truncated_bands]  # reverse occupied bands, so transition energy will from small to large
        empty_bands = empty_bands[:truncated_bands]

        transit_combo = list(itertools.product(occu_bands, empty_bands))

        for comb_ind, comb in enumerate(transit_combo):

            if isinstance(spin, Spin):
                pass
            elif isinstance(spin, str):
                spin = Spin(int(spin))

            if spin is Spin(1):
                spin_ind = 1
            else:
                spin_ind = 2

            transition_e, mu, tau = vaspwfc_analysis(ref_ind, kpoints, spin_ind, comb[0] + 1, comb[1] + 1,
                                                     wav_file, lgamma=lgamma)

            if spin not in tdm_tau_dict:
                tdm_tau_dict[spin] = {}

            tdm_tau_dict[spin].update({comb_ind: {"initial_band": comb[0],
                                                  "final_band": comb[1],
                                                  "ks_diff": transition_e,
                                                  "mu": mu,
                                                  "tau": tau}})

    return tdm_tau_dict
