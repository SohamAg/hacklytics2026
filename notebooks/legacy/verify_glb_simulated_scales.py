from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from pygltflib import GLTF2
except Exception:
    print('ERROR: pygltflib not installed. Install with: pip install pygltflib')
    raise

ROOT = Path('.').resolve()

# Try both data-processing and data_processing
clinical_path = ROOT / 'data-processing' / 'hvsmr_clinical.csv'
if not clinical_path.exists():
    clinical_path = ROOT / 'data_processing' / 'hvsmr_clinical.csv'

if not clinical_path.exists():
    raise FileNotFoundError(f'Could not find hvsmr_clinical.csv in data-processing or data_processing under {ROOT}')

hv = pd.read_csv(clinical_path)

with open(ROOT / 'models' / 'condition_effects.json', 'r', encoding='utf-8') as f:
    condition_effects = json.load(f)
cond_mult = condition_effects['condition_multipliers']

PART_NAME_TO_VOL_KEY = {
    'LV': 'Label_1_vol_ml','RV': 'Label_2_vol_ml','LA': 'Label_3_vol_ml','RA': 'Label_4_vol_ml',
    'AO': 'Label_5_vol_ml','PA': 'Label_6_vol_ml','PV': 'Label_7_vol_ml','SVC': 'Label_8_vol_ml',
}

GLB_MESH_TO_PART = {
    'Aorta_Aorta_0': 'AO',
    'Pulmonary_trunk_Pulmonary_trunk_0': 'PA',
    'Arteries1_Arteries1_0': 'PA',
    'Arteries3_Arteries3_0': 'PA',
    'Arteries5_Arteries5_0': 'PA',
    'Av_valves_Av_valves_0': None,
    'Valves5_Valves5_0': None,
    'Valves2_Valves2_0': None,
    'Ligament_Ligament_0': None,
    'Septum_Septum_0': None,
    'Heart_basis1_Heart_basis1_0': None,
    'Conduction__Conduction__0': None,
}

CORE_COLS = {'Pat','Age','Category'}
condition_cols = [c for c in hv.columns if c not in CORE_COLS]


def patient_conditions(pat_id:int):
    row = hv.loc[hv['Pat'].astype(int)==int(pat_id)].iloc[0]
    conds=[]
    for c in condition_cols:
        v=row[c]
        if isinstance(v,str) and v.strip().upper()=='X':
            conds.append(c)
    return conds


def compute_scales_for_conditions(selected_conditions, default=1.0):
    if not selected_conditions:
        return {part: float(default) for part in PART_NAME_TO_VOL_KEY.keys()}
    selected_conditions=[c for c in selected_conditions if c!='Normal']
    scales={part: float(default) for part in PART_NAME_TO_VOL_KEY.keys()}
    if not selected_conditions:
        return scales
    for part, vol_key in PART_NAME_TO_VOL_KEY.items():
        product=1.0; n=0
        for cond in selected_conditions:
            mult = cond_mult.get(cond, {}).get(vol_key, None)
            if mult is not None and mult>0:
                product *= float(mult); n+=1
        mult = (product**(1/n)) if n>0 else 1.0
        mult = max(0.2, min(5.0, mult))
        scales[part] = float(np.cbrt(mult))
    return scales


def _accessor_data(gltf: GLTF2, accessor_idx: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_idx]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    dtype_map = {5120: np.int8,5121: np.uint8,5122: np.int16,5123: np.uint16,5125: np.uint32,5126: np.float32}
    comp_dtype = dtype_map[accessor.componentType]
    type_num = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[accessor.type]
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_length = accessor.count * type_num * np.dtype(comp_dtype).itemsize
    data = gltf.binary_blob()
    arr = np.frombuffer(data[byte_offset:byte_offset+byte_length], dtype=comp_dtype)
    return arr.reshape((accessor.count, type_num))


def load_glb_meshes(glb_path: Path):
    gltf = GLTF2().load(str(glb_path))
    meshes = {}
    for mesh in gltf.meshes:
        name = mesh.name or 'UnnamedMesh'
        prim = mesh.primitives[0]
        verts = _accessor_data(gltf, prim.attributes.POSITION).astype(np.float32)
        meshes[name]=verts
    return meshes


def load_obj_groups(path: Path):
    verts=[]
    groups={}
    current='__ungrouped__'
    def ensure(g):
        if g not in groups:
            groups[g]=[]
    ensure(current)
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            if line.startswith('v '):
                parts=line.split(); verts.append([float(parts[1]),float(parts[2]),float(parts[3])])
            elif line.startswith('g '):
                current=line[2:].strip() or '__unnamed__'; ensure(current)
            elif line.startswith('f '):
                parts=line.split()[1:]
                idx=[int(p.split('/')[0])-1 for p in parts]
                if len(idx)>=3:
                    for k in range(1,len(idx)-1):
                        groups[current].append((idx[0], idx[k], idx[k+1]))
    verts=np.asarray(verts,dtype=np.float32)
    return verts, groups


def group_points(verts, faces):
    idx=np.unique(np.asarray(faces,dtype=np.int64).ravel()) if faces else np.array([],dtype=np.int64)
    return verts[idx] if idx.size>0 else np.zeros((0,3),dtype=np.float32)


def rms_radius(pts):
    if pts.shape[0]==0: return 0.0
    c=pts.mean(axis=0)
    return float(np.sqrt(((pts-c)**2).sum(axis=1).mean()))


base_meshes = load_glb_meshes(ROOT/'heart_models'/'cardiac_conduction_system.glb')

sim_dir = ROOT/'heart_models'/'simulated_patient_models'
if not sim_dir.exists():
    raise FileNotFoundError(f'Missing: {sim_dir}')

sim_files = sorted(sim_dir.glob('simulated_patient*.obj'))

TOL = 0.02
errors = 0

for sim_path in sim_files:
    # parse patient id from filename
    name = sim_path.stem
    try:
        pat_id = int(name.replace('simulated_patient',''))
    except Exception:
        continue

    verts, groups = load_obj_groups(sim_path)
    conds = patient_conditions(pat_id)
    scales = compute_scales_for_conditions(conds)

    for mesh_name, base_pts in base_meshes.items():
        part = GLB_MESH_TO_PART.get(mesh_name, None)
        expected = scales.get(part, 1.0) if part is not None else 1.0
        faces = groups.get(mesh_name, [])
        sim_pts = group_points(verts, faces)
        if sim_pts.shape[0]==0:
            continue
        r0 = rms_radius(base_pts)
        r1 = rms_radius(sim_pts)
        actual = (r1 / r0) if r0>0 else 1.0
        if abs(actual-expected) > TOL:
            errors += 1
            print(
                f'ERROR pat{pat_id} file={sim_path.name} mesh={mesh_name} '
                f'expected={expected:.3f} actual={actual:.3f}'
            )

if errors == 0:
    print('No scale mismatches found across all simulated models.')
else:
    print(f'Total mismatches: {errors}')
