from atomate.vasp.fireworks.hse0 import hse0FW
from atomate.vasp.config import VASP_CMD, DB_FILE, ADD_WF_METADATA
from atomate.vasp.powerups import add_modify_incar, add_wf_metadata, add_common_powerups, add_tags
from fireworks import Workflow
from monty.os.path import zpath
from pymatgen.core import Structure

def get_wf_hse0_continue(prev_calc_dir,
                         name="hse0_continue",
                         tags=None,
                         c=None,
                         hse0_incar_settings={}
                         ):

    c = c or {} # C is configurational dict
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    wf = []

    structure = Structure.from_file(zpath(prev_calc_dir + "/CONTCAR"))
    #structure = Structure.from_file('/global/homes/y/yyx5048/work/si_pbe_conv.POSCAR')

    fw = hse0FW(
        name=name,
        structure=structure,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        prev_calc_dir=prev_calc_dir,
        vasp_input_set_params= {'user_incar_settings':hse0_incar_settings}
    )

    wf = Workflow([fw])
    wf = add_common_powerups(wf, c)

    if isinstance(tags, list):
        wf = add_tags(wf, tags)

    return wf
