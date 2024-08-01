# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module provides ability to setup and parse defects on fly...
Note alot of this structure is taken from pycdt.utils.parse_calculations and pycdt.......

Requirements:
 - bulk calculation is finished (and vasprun.xml + Locpot)
 - a dielectric constant/tensor is provided
 - defect+chg calculation is finished 

 Soft requirements:
 	- Bulk and defect OUTCAR files (if charge correction by Kumagai et al. is desired)
 	- Hybrid bulk bandstructure / simple bulk structure calculation (if bandshifting is desired)
"""

import os
import itertools
import numpy as np

from monty.json import jsanitize

from pymatgen.io.vasp import Vasprun, Locpot, Poscar
from pymatgen import MPRester
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MVLScanRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.core import Vacancy, Substitution, Interstitial, ComplexMV, ComplexMI
from pymatgen.analysis.defects.generators import VacancyGenerator, SubstitutionGenerator, InterstitialGenerator, \
    VoronoiInterstitialGenerator, SimpleChargeGenerator, ComplexMVGenerator, ComplexMIGenerator
from pymatgen.analysis.structure_matcher import PointDefectComparator

from fireworks import FiretaskBase, FWAction, explicit_serialize, Workflow, Firework

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import TransmuterFW

from monty.serialization import dumpfn
from monty.json import MontyEncoder

from atomate.vasp.fireworks.hse0 import hse0FW

logger = get_logger(__name__)

def optimize_structure_sc_scale(inp_struct, final_site_no):
    """
    A function for finding optimal supercell transformation
    by maximizing the nearest image distance with the number of
    atoms remaining less than final_site_no

    Args:
        inp_struct: input pymatgen Structure object
        final_site_no (float or int): maximum number of atoms
        for final supercell

    Returns:
        3 x 1 array for supercell transformation
    """
    if final_site_no <= len(inp_struct.sites):
        final_site_no = len(inp_struct.sites)

    dictio={}
    #consider up to a 7x7x7 supercell
    for kset in itertools.product(range(1,7), range(1,7), range(1,7)):
        num_sites = len(inp_struct) * np.product(kset)
        if num_sites > final_site_no:
            continue

        struct = inp_struct.copy()
        struct.make_supercell(kset)

        #find closest image
        min_dist = 1000.
        for image_array in itertools.product( range(-1,2), range(-1,2), range(-1,2)):
            if image_array == (0,0,0):
                continue
            distance = struct.get_distance(0, 0, image_array)
            if distance < min_dist:
                min_dist = distance

        min_dist = round(min_dist, 3)
        if min_dist in dictio.keys():
            if dictio[min_dist]['num_sites'] > num_sites:
                dictio[min_dist].update( {'num_sites': num_sites, 'supercell': kset[:]})
        else:
            dictio[min_dist] = {'num_sites': num_sites, 'supercell': kset[:]}

    if not len(dictio.keys()):
        raise RuntimeError('could not find any supercell scaling vector')

    min_dist = max( list(dictio.keys()))
    biggest = dictio[ min_dist]['supercell']

    return biggest


@explicit_serialize
class DefectSetupFiretask(FiretaskBase):
    """
    Run defect supercell setup

    Args:
        structure (Structure): input structure to have defects run on
        cellmax (int): maximum supercell size to consider for supercells
        conventional (bool):
            flag to use conventional structure (rather than primitive) for supercells,
            defaults to True.
        vasp_cmd (string):
            the vasp cmd
        db_file (string):
            the db file
        user_incar_settings (dict):
            a dictionary of incar settings specified by user for both bulk and defect supercells
            note that charges do not need to be set in this dicitionary
        user_kpoints_settings (dict or Kpoints pmg object):
            a dictionary of kpoint settings specific by user OR an Actual Kpoint set to be used for the calculation

        vacancies (list):
            If list is totally empty, all vacancies are considered (default).
            If only specific vacancies are desired then add desired Element symbol to the list
                ex. ['Ga'] in GaAs structure will only produce Galium vacancies

            if NO vacancies are desired, then just add an empty list to the list
                ex. [ [] ]  yields no vacancies

        substitutions (dict):
            If dict is totally empty, all intrinsic antisites are considered (default).
            If only specific antisites/substituions are desired then add vacant site type as key, with list of
                sub site symbol as value
                    ex 1. {'Ga': ['As'] } in GaAs structure will only produce Arsenic_on_Gallium antisites
                    ex 2. {'Ga': ['Sb'] } in GaAs structure will only produce Antimonide_on_Gallium substitutions

            if NO antisites or substitutions are desired, then just add an empty dict
                ex. {'None':{}}  yields no antisites or subs

        interstitials (list):
            If list is totally empty, NO interstitial defects are considered (default).
            Option 1 for generation: If one wants to use Pymatgen to predict interstitial
                    then list of pairs of [symbol, generation method (str)] can be provided
                        ex. ['Ga', 'Voronoi'] in GaAs structure will produce Galium interstitials from the
                            Voronoi site finding algorithm
                        NOTE: only options for interstitial generation are "Voronoi" and "InFit"
            Option 2 for generation: If user wants to add their own interstitial sites for consideration
                    the list of pairs of [symbol, Interstitial object] can be provided, where the
                    Interstitial pymatgen.analysis.defects.core object is used to describe the defect of interest
                    NOTE: use great caution with this approach. You better be sure that the supercell with Interstitial in it
                        is same as the bulk supercell...

        initial_charges (dict):
            says how to specify initial charges for each defect.
            An empty dict (DEFAULT) is to do a fairly restrictive charge generation method:
                for vacancies: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try
                antisites and subs: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try +
                    added to likely charge of substitutional site (closest to zero)
                interstitial: charge zero
            For non empty dict, charges are specified as:
                initial_charges = {'vacancies': {'Ga': [-3,2,1,0]},
                                   'substitutions': {'Ga': {'As': [0]} },
                                   'interstitials': {}}
                in the GaAs structure this makes vacancy charges in states -3,-2,-1,0; Ga_As antisites in the q=0 state,
                and all other defects will have charges generated in the restrictive automated format stated for DEFAULT

    """
    def run_task(self, fw_spec):
        if os.path.exists("POSCAR"):
            structure =  Poscar.from_file("POSCAR").structure
        else:
            structure = self.get("structure")

        if self.get("conventional", True):
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        fws, parents = [], []

        cellmax=self.get("cellmax", 128)
        sc_scale = optimize_structure_sc_scale(structure, cellmax)
        # TODO: hard coded here for 2d, we need orthogonal cell
        sc_mat = [6,4,1] * np.identity(3)
        trans_mat = np.array([[1,1,0],[-1,1,0],[0,0,1]])
        sc_scale = np.dot(sc_mat,trans_mat)

        #First Firework is for bulk supercell
        bulk_supercell = structure.copy()
        bulk_supercell.make_supercell(sc_scale)
        num_atoms = len(bulk_supercell)

        user_incar_settings = self.get("user_incar_settings", {})
        user_kpoints_settings = self.get("user_kpoints_settings", {})

        print(f"user_incar_settings setting is {user_incar_settings}")

        bulk_incar_settings = {"EDIFF":.0001, "EDIFFG": 0.001, "ISMEAR":0, "SIGMA":0.05, "NSW": 0, "ISIF": 2,
                               "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "LWAVE": True}
        bulk_incar_settings.update( user_incar_settings)

        reciprocal_density = 100
        kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": reciprocal_density}
        vis = MPRelaxSet( bulk_supercell,
                          user_incar_settings=bulk_incar_settings,
                          user_kpoints_settings=kpoints_settings)

        # TODO: hard coded the supercell_size for 2d
        #supercell_size = sc_scale * np.identity(3) # this is 3d
        supercell_size = sc_scale # this is 2d
        bulk_tag = "{}:bulk_supercell_{}atoms".format(structure.composition.reduced_formula, num_atoms)
        stat_fw = TransmuterFW(name = bulk_tag, structure=structure,
                               transformations=['SupercellTransformation'],
                               transformation_params=[{"scaling_matrix": supercell_size}],
                               vasp_input_set=vis, copy_vasp_outputs=False,
                               vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                               db_file=self.get("db_file", ">>db_file<<"))

        fws.append(stat_fw)

        # make defect set
        vacancies = self.get("vacancies", list())
        substitutions = self.get("substitutions", dict())
        interstitials = self.get("interstitials", list())
        initial_charges  = self.get("initial_charges", dict())

        complex_defect = self.get("complex_defect", list()) # TODO: I am not sure about this.. should it be a list?
        complex_defect_only = self.get("complex_defect_only", False)

        def_structs = []
        # a list with following dict structure for each entry:
        # {'defect': pymatgen defect object type,
        # 'charges': list of charges to run}

        if not vacancies:
            # default: generate all vacancies...
            b_struct = structure.copy()
            VG = VacancyGenerator( b_struct)
            for vac_ind, vac in enumerate(VG):
                vac_symbol = vac.site.specie.symbol

                charges = []
                if initial_charges:
                    if 'vacancies' in initial_charges.keys():
                        if vac_symbol in initial_charges['vacancies']:
                            #NOTE if more than one type of vacancy for a given specie, this will assign same charges to all
                            charges = initial_charges['vacancies'][vac_symbol]

                if not len(charges):
                    SCG = SimpleChargeGenerator(vac.copy())
                    charges = [v.charge for v in SCG]

                def_structs.append({'charges': charges, 'defect': vac.copy()})
        else:
            # only create vacancies of interest...
            for elt_type in vacancies:
                b_struct = structure.copy()
                VG = VacancyGenerator( b_struct)
                for vac_ind, vac in enumerate(VG):
                    vac_symbol = vac.site.specie.symbol
                    if elt_type != vac_symbol:
                        continue

                    charges = []
                    if initial_charges:
                        if 'vacancies' in initial_charges.keys():
                            if vac_symbol in initial_charges['vacancies']:
                                # NOTE if more than one type of vacancy for a given specie,
                                # this will assign same charges to all
                                charges = initial_charges['vacancies'][vac_symbol]

                    if not len(charges):
                        SCG = SimpleChargeGenerator(vac.copy())
                        charges = [v.charge for v in SCG]

                    def_structs.append({'charges': charges, 'defect': vac.copy()})

        if not substitutions:
            # default: set up all intrinsic antisites
            for sub_symbol in [elt.symbol for elt in bulk_supercell.types_of_specie]:
                b_struct = structure.copy()
                SG = SubstitutionGenerator(b_struct, sub_symbol)
                for as_ind, sub in enumerate(SG):
                    # find vac_symbol to correctly label defect
                    poss_deflist = sorted(sub.bulk_structure.get_sites_in_sphere(sub.site.coords, 2, include_index=True), key=lambda x: x[1])
                    defindex = poss_deflist[0][2]
                    vac_symbol = sub.bulk_structure[defindex].specie.symbol

                    charges = []
                    if initial_charges:
                        if 'substitutions' in initial_charges.keys():
                            if vac_symbol in initial_charges['substitutions']:
                                # NOTE if more than one type of substituion for a given specie,
                                # this will assign same charges to all
                                if sub_symbol in initial_charges['substitutions'][vac_symbol].keys():
                                    charges = initial_charges['substitutions'][vac_symbol][sub_symbol]
                    if not len(charges):
                        SCG = SimpleChargeGenerator(sub.copy())
                        charges = [v.charge for v in SCG]

                    def_structs.append({'charges': charges, 'defect': sub.copy()})
        else:
            # only set up specified antisite / substituion types
            for vac_symbol, sub_list in substitutions.items():
                for sub_symbol in sub_list:
                    b_struct = structure.copy()
                    SG = SubstitutionGenerator(b_struct, sub_symbol)
                    for as_ind, sub in enumerate(SG):
                        # find vac_symbol for this sub defect
                        poss_deflist = sorted(sub.bulk_structure.get_sites_in_sphere(sub.site.coords, 2, include_index=True), key=lambda x: x[1])
                        defindex = poss_deflist[0][2]
                        gen_vac_symbol = sub.bulk_structure[defindex].specie.symbol
                        if vac_symbol != gen_vac_symbol: # only consider subs on specfied vac_symbol site
                            continue

                        charges = []
                        if initial_charges:
                            if 'substitutions' in initial_charges.keys():
                                if vac_symbol in initial_charges['substitutions']:
                                    # NOTE if more than one type of substituion for a given specie,
                                    # this will assign same charges to all
                                    if sub_symbol in initial_charges['substitutions'][vac_symbol].keys():
                                        charges = initial_charges['substitutions'][vac_symbol][sub_symbol]
                        if not len(charges):
                            SCG = SimpleChargeGenerator(sub.copy())
                            charges = [v.charge for v in SCG]

                        def_structs.append({'charges': charges, 'defect': sub.copy()})

        if interstitials:
            # default = do not include interstitial defects

            def get_charges_from_inter( inter_elt):
                inter_charges = []
                if initial_charges:
                    if 'interstitials' in initial_charges.keys():
                        if inter_elt in initial_charges['interstitials']:
                            # NOTE if more than one type of interstitial for a given specie,
                            # this will assign same charges to all
                            inter_charges = initial_charges['interstitials'][inter_elt]

                if not len(inter_charges):
                    SCG = SimpleChargeGenerator(inter_elt)
                    inter_charges = [v.charge for v in SCG]
                return inter_charges

            for elt_type, elt_val in interstitials:
                if type(elt_val) == str:
                    b_struct = structure.copy()
                    if elt_val == 'Voronoi':
                        IG = VoronoiInterstitialGenerator(b_struct, elt_type)
                    elif elt_val == 'InFit':
                        IG = InterstitialGenerator(b_struct, elt_type)
                    else:
                        raise ValueError('Interstitial finding method not recognized. '
                                         'Please choose either Voronoi or InFit.')

                    for inter_ind, inter in enumerate(IG):
                        charges = get_charges_from_inter( elt_type)
                        def_structs.append({'charges': charges, 'defect': inter.copy()})
                else:
                    charges = get_charges_from_inter( elt_val)
                    def_structs.append({'charges': charges, 'defect': elt_val.copy()})

        if complex_defect: #TODO: Using existing entries
            comp_def_structs = []

            if "MV" in complex_defect:

                b_struct = structure.copy()
                sub_def_structs = []
                vac_def_structs = []

                for simple_defcalc in def_structs:
                    if isinstance(simple_defcalc['defect'], Substitution):
                        sub_def_structs.append(simple_defcalc)

                    if isinstance(simple_defcalc['defect'], Vacancy):
                        vac_def_structs.append(simple_defcalc)

                if not (vac_def_structs and sub_def_structs):
                    raise ValueError('Simple vacancies or substitutions are not created.'
                                     'This is required from MV complex.')

                pdc = PointDefectComparator(check_charge=False, check_primitive_cell=False, check_lattice_scale=False)
                for sub in sub_def_structs:
                    MVG = ComplexMVGenerator(b_struct, sub['defect'].site.species_string)

                    for mv_ind, mv_def in enumerate(MVG):
                        if pdc.are_equal(mv_def.substitution, sub['defect']):
                            # check vacancy for comparison
                            for vac in vac_def_structs:
                                if pdc.are_equal(mv_def.vacancy, vac['defect']):
                                    mv_chg = itertools.product(sub['charges'], vac['charges'])
                                    mv_charges = list(set([sum(chg) for chg in mv_chg]))

                                    comp_def_structs.append({'charges': mv_charges, 'defect': mv_def.copy()})

            if "MI" in complex_defect:

                sub_def_structs = []
                inter_def_structs = []

                for simple_defcalc in def_structs:
                    if isinstance(simple_defcalc['defect'], Substitution):
                        sub_def_structs.append(simple_defcalc)

                    if isinstance(simple_defcalc['defect'], Interstitial):
                        inter_def_structs.append(simple_defcalc)

                if not (sub_def_structs and inter_def_structs):
                    raise ValueError('Simple vacancies or interstitials are not created.'
                                     'This is required from MI complex.')
                counter=0
                for sub in sub_def_structs:
                    for inter in inter_def_structs:
                        if (inter['defect'].site.specie != sub['defect'].site.specie) and (inter['defect'].site.specie != 'Si'):
                            counter+=1
                            print(f"{counter} defect created: Int_{inter['defect'].site.specie} + Sub_{sub['defect'].site.specie}")
                            mi_def = ComplexMI(sub['defect'], inter['defect'])
                            mi_chg = itertools.product(sub['charges'], inter['charges'])
                            mi_charges = list(set([sum(chg) for chg in mi_chg]))
                            # extend the charge to complete range:
                            mi_charges = list(
                                range(
                                    min(min(mi_charges), 0), max(max(mi_charges), 0) + 1
                                ))

                            comp_def_structs.append({'charges': mi_charges, 'defect': mi_def.copy()})

                        # TODO: these are the original codes
                        #mi_def = ComplexMI(sub['defect'], inter['defect'])
                        #mi_chg = itertools.product(sub['charges'], inter['charges'])
                        #mi_charges = list(set([sum(chg) for chg in mi_chg]))
                        #comp_def_structs.append({'charges': mi_charges, 'defect': mi_def.copy()})

            if not comp_def_structs:  # no complex defect generated fit the criteria
                raise ValueError("No complex defect match the simple defect")

            if complex_defect_only:
                def_structs = comp_def_structs
            else:
                def_structs = def_structs + comp_def_structs

        # now that def_structs is assembled, set up Transformation FW for all defect + charge combinations
        hse0 = self.get("hse0", False)
        links = {}

        for defcalc in def_structs:
            #if defcalc['defect'].interstitial.multiplicity != 1:
            #    continue
            #iterate over all charges to be run
            for charge in defcalc['charges']:
                #if charge == -1:
                #    continue
                defect = defcalc['defect'].copy()
                defect.set_charge(charge)

                # TODO: can we add tag for the defect calculation?
                qis_data_dict = self.get("qis_data_dict", {})
                fw = get_fw_from_defect( defect, supercell_size,
                                         user_kpoints_settings = user_kpoints_settings,
                                         user_incar_settings = user_incar_settings,
                                         qis_data_dict = qis_data_dict,
                                         vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                                         )

                fws.append(fw)

                # append HSE0 calculations
                # TODO: think about the names, can we pass the tags along?
                # TODO: think about how to add tags?
                if hse0:
                    if complex_defect:
                        tags = [defect.name.split('_')[0], 'hse0']
                    else:
                        tags = ['hse0']

                    hse0_incar_settings = {"user_incar_settings": self.get("hse0_incar_settings", {})}
                    print(f"hse0_incar_settings in DefectSetupFiretask: {hse0_incar_settings}")

                    bulk_sc = defect.bulk_structure.copy()
                    bulk_sc.make_supercell(supercell_size)
                    num_atoms = len(bulk_sc)
                    hse0_name = "{}:hse0_{}_{}_{}atoms".format(defect.bulk_structure.composition.reduced_formula,
                                                          defect.name, defect.charge, num_atoms)

                    hse0_fw = hse0FW(
                        structure = defect.bulk_structure,
                        name=hse0_name,
                        parents=fw,
                        vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                        db_file=self.get("db_file", ">>db_file<<"),
                        vasp_input_set_params=hse0_incar_settings,
                        tags=tags,
                    )
                    # add a tag here
                    fws.append(hse0_fw)
                    links.update({fw: [hse0_fw]})
        if hse0:
            wf = Workflow(fws, links)
            return FWAction(detours=wf)  # maintain the child parent relationship
        else:
            return FWAction(detours=fws)


def get_fw_from_defect( defect, supercell_size, user_kpoints_settings = {},
                        user_incar_settings = {},
                        db_file='>>db_file<<', vasp_cmd='>>vasp_cmd<<', qis_data_dict={}):
    """
    Simple function for grabbing fireworks for a defect, given a supercell_size
    :param defect:
    :param supercell_size:
    :param user_kpoints_settings:
    :param db_file:
    :param vasp_cmd:
    :return:
    """
    chgd_sc_struct = defect.generate_defect_structure(supercell=supercell_size)

    reciprocal_density = 100

    kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": reciprocal_density}

    # NOTE that the charge will be reflected in NELECT of INCAR because use_structure_charge=True
    stdrd_defect_incar_settings = {"EDIFF": 0.0001, "EDIFFG": 0.001, "IBRION": 2, "ISMEAR": 0, "SIGMA": 0.05,
                                   "ISPIN": 2, "ISYM": 0, "LVHAR": True, "LVTOT": True, "NSW": 100,
                                   "NELM": 60, "ISIF": 2, "LAECHG": False, "LWAVE": True}
    stdrd_defect_incar_settings.update(user_incar_settings)
    defect_input_set = MPRelaxSet(chgd_sc_struct,
                                  user_incar_settings=stdrd_defect_incar_settings.copy(),
                                  user_kpoints_settings=kpoints_settings,
                                  use_structure_charge=True)

    # get defect site for parsing purposes
    # TODO: comment out this portion, should we do wavecar localization any more?
    #struct_for_defect_site = Structure(defect.bulk_structure.copy().lattice,
    #                                   [defect.site.specie],
    #                                   [defect.site.frac_coords],
    #                                   to_unit_cell=True, coords_are_cartesian=False)
    #struct_for_defect_site.make_supercell(supercell_size)
    #defect_site = struct_for_defect_site[0]

    bulk_sc = defect.bulk_structure.copy()
    bulk_sc.make_supercell( supercell_size)
    num_atoms = len(bulk_sc)

    chgdef_trans = ["DefectTransformation"]
    chgdef_trans_params = [{"scaling_matrix": supercell_size,
                            "defect": defect.copy()}]

    def_tag = "{}:{}_{}_{}atoms".format(defect.bulk_structure.composition.reduced_formula,
                                        defect.name, defect.charge, num_atoms)
    fw = TransmuterFW(name=def_tag, structure=defect.bulk_structure,
                      transformations=chgdef_trans,
                      transformation_params=chgdef_trans_params,
                      vasp_input_set=defect_input_set,
                      vasp_cmd=vasp_cmd,
                      copy_vasp_outputs=False,
                      db_file=db_file,
                      bandstructure_mode="auto",
                      defect_wf_parsing=True,
                      qis_data_dict=qis_data_dict
                      #defect_wf_parsing=defect_site
                      )

    return fw

