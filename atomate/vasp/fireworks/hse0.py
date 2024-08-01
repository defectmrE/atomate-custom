from fireworks import Firework

from pymatgen.io.vasp.sets import MPHSE0StaticSet
from atomate.common.firetasks.glue_tasks import (
    PassCalcLocs,
    CopyFiles,
    DeleteFiles,
    CreateFolder,
)
from atomate.vasp.firetasks import (
    CopyVaspOutputs,
    RunVaspCustodian,
    VaspToDb,
)

from atomate.vasp.firetasks.write_inputs import WriteVaspHSE0FromPrev


class hse0FW(Firework):

    def __init__(self, parents=None, prev_calc_dir=None, structure=None, name="hse0_Static",
                 vasp_input_set_params=None, tags=None, vasp_cmd=None, db_file=None, vasptodb_kwargs=None,
                 **kwargs):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = fw_name

        user_incar_settings = vasp_input_set_params.get("user_incar_settings", {})

        if prev_calc_dir:
            t.append(WriteVaspHSE0FromPrev(prev_calc_dir=prev_calc_dir,
                                           copy_wavecar=True,
                                           copy_chgcar=True,
                                           use_structure_charge=True,
                                           other_params={"user_incar_settings": user_incar_settings},
                                           **vasp_input_set_params))

        elif parents:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True, additional_files=["CHGCAR","WAVECAR"]))
            #t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True,additional_files=["CHGCAR", "WAVECAR"]))
            t.append(WriteVaspHSE0FromPrev(prev_calc_dir=".",
                                           copy_wavecar=False,
                                           copy_chgcar=False,
                                           use_structure_charge=True,
                                           other_params={"user_incar_settings": user_incar_settings},
                                           **vasp_input_set_params))

        elif structure:
            raise ValueError("HSE0 calculation from scratch has not been implemented yet.")

        else:
            raise ValueError("Must specify previous calculation to perform HSE0 calculations")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, handler_group="no_handler", auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        if tags:
            vasptodb_kwargs["additional_fields"]["tags"] = tags
        t.append(
            VaspToDb(db_file=db_file, parse_eigenval=True, **vasptodb_kwargs))
        super().__init__(t, parents=parents, name=fw_name, **kwargs)
