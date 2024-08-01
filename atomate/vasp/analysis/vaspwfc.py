import numpy as np
import scipy.constants as const

from pymatgen.electronic_structure.core import Spin


def vaspwfc_analysis(ref_ind, kpoints, spin, i_band, f_band, wav_file, lgamma):
    """
    compute transition dipole moment and lifetime
    Args:
        ref_ind: refractive index
        kpoints: kpoint index (int)
        spin: spin index, 1 for up, 2 for down (int)
        i_band: initial band index (int, list or numpy array)
        f_band: final band index (int, list or numpy array), starts from 1
        wave_file: WAVECAR file name (str)
        lgamma: If it is calcuated using vasp_gam (boolean)


    Returns:

    """
    from vaspwfc import vaspwfc

    # check types:
    if isinstance(i_band, list):
        i_band = np.asarray(i_band)
    if isinstance(i_band, int):
        i_band = np.asarray([i_band])

    if isinstance(f_band, list):
        f_band = np.asarray(f_band)
    if isinstance(f_band, int):
        f_band = np.asarray([f_band])

    epsilon0 = const.epsilon_0  # vacuum permittivity
    h = const.Planck
    c = const.speed_of_light
    eV_to_Hz = const.physical_constants['electron volt-hertz relationship'][0]

    wav = vaspwfc(wav_file, lgamma=lgamma)

    band_ind = []
    transition_e_list = []
    mu_list = []
    tau_list = []
    #print(i_band)
    #print(f_band)

    for ib in i_band:
        for fb in f_band:
            if fb > ib:
                res = wav.TransitionDipoleMoment([spin, kpoints, ib + 1],
                                                 [spin, kpoints, fb + 1],
                                                 norm=True)
                mu = np.linalg.norm(res[-1])
                musq = mu ** 2

                transition_e = res[2]
                freq = transition_e * eV_to_Hz
                A = (2 * ref_ind * (2 * const.pi) ** 3 * freq ** 3 * musq * (3.33e-30) ** 2) / (
                        3 * epsilon0 * h * c ** 3)  # The 3.33e-30 converts to Debye
                tau = 1 / A * 1e9  # in unit of ns

                band_ind.append([ib, fb])
                transition_e_list.append(transition_e)
                mu_list.append(mu)
                tau_list.append(tau)

    return band_ind, transition_e_list, mu_list, tau_list


def get_all_tdm_and_lifetime(ref_ind, kpoints, wav_file, lgamma, eigenval, vbm, cbm, emin=-1, emax=1,
                             truncated_bands=20, all_combination=False):
    """

    Args:
        ref_ind: refractive index of the materials (float)
        kpoints: kpoints index, gamma point is 1 (int)
        wav_file: WAVECAR file name (str)d
        lgamma: If it is calcuated using vasp_gam (boolean)
        int_eigenval: integerized eigenvalue files
        vbm: valence band maximum in eV
        cbm: conduction band minimum in eV
        emin: lowerbound energy below VBM
        emax: upperbound energy above CBM
        truncated_bands: limit occupied states and unoccupied states, default to 20

    Returns:
        dictionary contains exitation order, initial and final state band index, mu and tau.
    """

    tdm_tau_dict = {}

    for spin, egv in eigenval.items():

        assert egv.shape[0] == 1, f"Only supports gamma point calculations!"

        homo_band_ind = np.where(egv[0, :, 1] < 1e-8)[0][0] - 1

        band_range = np.where((vbm + emin < egv[0, :, 0]) & (egv[0, :, 0] < cbm + emax))[0]

        occu_bands = []
        empty_bands = []

        for band_ind in band_range:
            if band_ind <= homo_band_ind:
                occu_bands.append(band_ind)
            else:
                empty_bands.append(band_ind)

        occu_bands = occu_bands[-truncated_bands:]
        empty_bands = empty_bands[:truncated_bands]

        if all_combination:
            occu_bands = occu_bands + empty_bands
            empty_bands = occu_bands.copy()


        if isinstance(spin, Spin):
            pass
        elif isinstance(spin, str):
            spin = Spin(int(spin))

        if spin is Spin(1):
            spin_ind = 1
        else:
            spin_ind = 2

        band_combo, transition_e, mu, tau = vaspwfc_analysis(ref_ind, kpoints, spin_ind, occu_bands, empty_bands,
                                                             wav_file, lgamma=lgamma)

        if spin not in tdm_tau_dict:
            tdm_tau_dict[spin] = {}

        sorted_excitation = sorted(zip(band_combo, transition_e, mu, tau), key=lambda x: x[1])  # sort based on energy
        for comb_ind, d in enumerate(sorted_excitation):
            tdm_tau_dict[spin].update({comb_ind: {"initial_band": d[0][0],
                                                  "final_band": d[0][1],
                                                  "ks_diff": d[1],
                                                  "mu": d[2],
                                                  "tau": d[3]}})

    return tdm_tau_dict
