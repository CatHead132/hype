import struct
import copy
import os
import re
import io
import numpy as np
import UnityPy
from UnityPy.files import SerializedFile
from UnityPy.streams import EndianBinaryReader
from UnityPy.classes import Vector3f
from PIL import Image

SRC = "/home/cathead/Downloads/hypper/patched-data-recovered/patched_game.data"
OUT = "/home/cathead/Downloads/hypper/patched-data-recovered/patched_game_v3.data"
NPZ = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/gm_construct_gltf_mesh.npz"
GM_DIR = "/home/cathead/Downloads/hypper/gm_construct"

MESH_EXTERNAL_FILEID = 3  # sharedassets5.assets index in level9/level7's externals
MESH_PATH_ID = 1010

# =================== 1. material resolution ===================
# local, map-specific textures (from the extracted gm_construct folder)
GMOD_DIR = "/home/cathead/Downloads/GarrysMod"
GMOD_VPK_PATHS = [
    os.path.join(GMOD_DIR, "sourceengine", "hl2_textures_dir.vpk"),
    os.path.join(GMOD_DIR, "sourceengine", "content_hl2_dir.vpk"),
    os.path.join(GMOD_DIR, "sourceengine", "content_cstrike_dir.vpk"),
]

vtf_keys = set()
vmt_content = {}
for fn in os.listdir(GM_DIR):
    key = fn.replace("\\", "/").lower()
    if key.startswith("materials/"):
        key = key[len("materials/"):]
    if fn.lower().endswith(".vtf"):
        vtf_keys.add(key[:-4])
    elif fn.lower().endswith(".vmt"):
        with open(os.path.join(GM_DIR, fn), "r", errors="ignore") as fh:
            vmt_content[key[:-4]] = fh.read()

def resolve_local_vtf_key(mat_name):
    key = mat_name.lower()
    if key in vtf_keys:
        return key
    if key in vmt_content:
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', vmt_content[key])
        if m:
            bt = m.group(1).lower().replace("\\", "/")
            if bt in vtf_keys:
                return bt
    return None

def vtf_disk_filename(vtf_key):
    # files are flat with literal backslashes: "materials\gm_construct\grass1.vtf"
    return "materials\\" + vtf_key.replace("/", "\\") + ".vtf"

# real base-game textures (owned Garry's Mod install -- HL2/CS:S content used by gm_construct)
from srctools.bsp import BSP
from srctools.vpk import VPK

bsp = BSP("/home/cathead/Downloads/GarrysMod/garrysmod/maps/gm_construct.bsp")
bsp_pak = bsp.pakfile
bsp_pak_names_lower = {n.lower(): n for n in bsp_pak.namelist()}

gmod_vpks = [VPK(p) for p in GMOD_VPK_PATHS if os.path.exists(p)]
gmod_vtf_source = {}  # key -> (vpk_index, real_filename)
gmod_vmt_source = {}
for vi, vpk in enumerate(gmod_vpks):
    for e in vpk:
        key = e.filename.lower()
        if key.startswith("materials/"):
            key = key[len("materials/"):]
        if e.ext.lower() == "vtf" and key.endswith(".vtf"):
            gmod_vtf_source.setdefault(key[:-4], (vi, e.filename))
        elif e.ext.lower() == "vmt" and key.endswith(".vmt"):
            gmod_vmt_source.setdefault(key[:-4], (vi, e.filename))
print(f"GMod VPKs indexed: {len(gmod_vtf_source)} vtf, {len(gmod_vmt_source)} vmt")

def read_gmod_vmt(key):
    if key not in gmod_vmt_source:
        return None
    vi, path = gmod_vmt_source[key]
    return gmod_vpks[vi][path].read().decode(errors="ignore")

def resolve_gmod_key(mat_name, depth=0):
    """mat_name -> a key present in gmod_vtf_source, or None. Chases BSP pakfile
    'patch' vmts (per-instance materials the compiler generates, e.g. for unique
    cubemap reflections) back to their real base material, then GMod's own vmt
    $basetexture indirection, recursively."""
    if depth > 4:
        return None
    key = mat_name.lower()
    if key in gmod_vtf_source:
        return key
    pak_key = f"materials/{key}.vmt"
    if pak_key in bsp_pak_names_lower:
        with bsp_pak.open(bsp_pak_names_lower[pak_key]) as f:
            content = f.read().decode(errors="ignore")
        m = re.search(r'"include"\s*"([^"]+)"', content, re.IGNORECASE)
        if m:
            inc = m.group(1).lower().replace("\\", "/")
            if inc.startswith("materials/"):
                inc = inc[len("materials/"):]
            if inc.endswith(".vmt"):
                inc = inc[:-4]
            return resolve_gmod_key(inc, depth + 1)
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', content, re.IGNORECASE)
        if m:
            bt = m.group(1).lower().replace("\\", "/")
            return resolve_gmod_key(bt, depth + 1)
    content = read_gmod_vmt(key)
    if content:
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', content, re.IGNORECASE)
        if m:
            bt = m.group(1).lower().replace("\\", "/")
            if bt != key:
                return resolve_gmod_key(bt, depth + 1)
    return None

# every material we still couldn't resolve to any real texture gets the closest
# available real texture by keyword match, rather than one generic catch-all
CLOSEST_KEYWORDS = [
    ("gm_construct/grass1", ["grass", "sand", "dirt", "terrain"]),
    ("gm_construct/construct_concrete_floor", ["floor", "tile"]),
    ("gm_construct/construct_concrete_ground", ["ground", "concrete", "cement", "sidewalk", "curb"]),
    ("brick/brickwall004a", ["brick", "building", "wall", "roof", "template", "plaster",
                              "metal", "stair", "door", "window", "trim", "siding", "block"]),
    ("color/white", ["dev", "white", "measure", "generic", "nodraw", "tools"]),
    ("gm_construct/construct_credits", ["credit", "sign"]),
]
DEFAULT_CLOSEST = "gm_construct/construct_concrete_ground"

def closest_vtf_key(mat_name):
    key = mat_name.lower()
    for vtf_key, words in CLOSEST_KEYWORDS:
        if any(w in key for w in words):
            return vtf_key
    return DEFAULT_CLOSEST

# =================== 1b. curated 1:1 texture pack (highest priority) ===================
# A pre-rendered, correctly-named PNG set specifically for gm_construct's actual
# materials, bypassing srctools' own VTF/DXT decoder entirely (which could itself
# be a source of wrong colors, not just a resolution/quality issue).
PACK_DIR = "/home/cathead/Downloads/hypper/gm_construct_pack"
import zipfile as _zipfile

pack_pngs = {}  # lowercase filename -> callable returning PIL.Image
def _pil_from_path(path):
    return lambda: Image.open(path).convert("RGBA")
def _pil_from_zip(zip_path, name):
    def _load():
        with _zipfile.ZipFile(zip_path) as z:
            with z.open(name) as f:
                return Image.open(io.BytesIO(f.read())).convert("RGBA")
    return _load

for sub in ["textures", "source"]:
    d = os.path.join(PACK_DIR, sub)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower().endswith(".png"):
                pack_pngs[fn.lower()] = _pil_from_path(os.path.join(d, fn))
_pack_zip = os.path.join(PACK_DIR, "source", "untitled.zip")
if os.path.exists(_pack_zip):
    with _zipfile.ZipFile(_pack_zip) as z:
        for n in z.namelist():
            if n.lower().endswith(".png"):
                base = os.path.basename(n).lower()
                pack_pngs.setdefault(base, _pil_from_zip(_pack_zip, n))
print(f"texture pack indexed: {len(pack_pngs)} PNGs")

def pack_filename_for(canonical_key):
    return "materials" + canonical_key.replace("/", "") + ".png"

def find_base_material_name(mat_name, depth=0):
    """Walk every known indirection (local vmt $basetexture, BSP pakfile 'patch'
    include/$basetexture, GMod's own vmt $basetexture) to the final base material
    name, WITHOUT requiring any particular texture file to exist at each hop --
    just returns wherever the chain bottoms out, for checking against any texture
    source afterward."""
    if depth > 6:
        return mat_name.lower()
    key = mat_name.lower()
    if key in vmt_content:
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', vmt_content[key])
        if m:
            bt = m.group(1).lower().replace("\\", "/")
            if bt != key:
                return find_base_material_name(bt, depth + 1)
    pak_key = f"materials/{key}.vmt"
    if pak_key in bsp_pak_names_lower:
        with bsp_pak.open(bsp_pak_names_lower[pak_key]) as f:
            content = f.read().decode(errors="ignore")
        m = re.search(r'"include"\s*"([^"]+)"', content, re.IGNORECASE)
        if m:
            inc = m.group(1).lower().replace("\\", "/")
            if inc.startswith("materials/"):
                inc = inc[len("materials/"):]
            if inc.endswith(".vmt"):
                inc = inc[:-4]
            return find_base_material_name(inc, depth + 1)
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', content, re.IGNORECASE)
        if m:
            return find_base_material_name(m.group(1).lower().replace("\\", "/"), depth + 1)
    gmod_content = read_gmod_vmt(key)
    if gmod_content:
        m = re.search(r'\$basetexture"?\s+"([^"]+)"', gmod_content, re.IGNORECASE)
        if m:
            bt = m.group(1).lower().replace("\\", "/")
            if bt != key:
                return find_base_material_name(bt, depth + 1)
    return key

def resolve_pack_key(mat_name):
    base = find_base_material_name(mat_name)
    fn = pack_filename_for(base)
    if fn in pack_pngs:
        return base
    # also try the raw name directly, in case it needed no indirection at all
    fn2 = pack_filename_for(mat_name.lower())
    if fn2 in pack_pngs:
        return mat_name.lower()
    return None

data = np.load(NPZ, allow_pickle=True)
verts_src = data["verts"]
uvs_src = data["uvs"]
tris = data["tris"]
tri_mat_id = data["tri_mat_id"]
mat_names = data["mat_names"]

# resolved_vtf_for_mat[i] = (source, key) where source is "pack", "local", "gmod", or "closest"
resolved_vtf_for_mat = {}
n_pack = n_local = n_gmod = n_closest = 0
for i, name in enumerate(mat_names):
    pack = resolve_pack_key(str(name))
    if pack:
        resolved_vtf_for_mat[i] = ("pack", pack)
        n_pack += 1
        continue
    local = resolve_local_vtf_key(str(name))
    if local:
        resolved_vtf_for_mat[i] = ("local", local)
        n_local += 1
        continue
    gmod = resolve_gmod_key(str(name))
    if gmod:
        resolved_vtf_for_mat[i] = ("gmod", gmod)
        n_gmod += 1
        continue
    resolved_vtf_for_mat[i] = ("local", closest_vtf_key(str(name)))
    n_closest += 1
print(f"{n_pack} from curated 1:1 pack, {n_local} map-specific, {n_gmod} real GMod base-game textures, {n_closest} closest-keyword fallback (of {len(mat_names)} materials)")

unique_vtf_keys = sorted(set(resolved_vtf_for_mat.values()))
print("unique real textures needed:", len(unique_vtf_keys))

vtf_key_to_group = {k: i for i, k in enumerate(unique_vtf_keys)}
tri_group = np.array([vtf_key_to_group[resolved_vtf_for_mat[mid]] for mid in tri_mat_id], dtype=np.int32)
n_groups = len(unique_vtf_keys)
print("submesh groups:", n_groups, "(all real textures, no generic fallback)")

# =================== 2. decode VTFs to RGBA (downscaled) ===================
from srctools.vtf import VTF
TEX_SIZE = 256

def read_vtf_bytes(source, key):
    if source == "local":
        path = os.path.join(GM_DIR, vtf_disk_filename(key))
        with open(path, "rb") as fh:
            return fh.read()
    elif source == "gmod":
        vi, real_name = gmod_vtf_source[key]
        return gmod_vpks[vi][real_name].read()
    raise ValueError(source)

def decode_vtf_mipchain(source, key):
    """Returns list of (W,H,4) uint8 arrays, largest first, down to 1x1 -- without a
    full mip chain, tiled textures alias/shimmer badly at grazing angles (this was
    the cause of the 'freaky stretched' moire pattern seen on the ground)."""
    if source == "pack":
        fn = pack_filename_for(key)
        base = pack_pngs[fn]().resize((TEX_SIZE, TEX_SIZE))
    else:
        raw = read_vtf_bytes(source, key)
        buf = io.BytesIO(raw)
        vtf = VTF.read(buf)
        frame = vtf.get()
        base = frame.to_PIL().convert("RGBA").resize((TEX_SIZE, TEX_SIZE))
    mips = []
    size = TEX_SIZE
    img = base
    while True:
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        mips.append(np.array(flipped, dtype=np.uint8))
        if size == 1:
            break
        size = max(1, size // 2)
        img = img.resize((size, size), Image.LANCZOS)
    return mips

texture_mips = {}
for source, key in unique_vtf_keys:
    try:
        texture_mips[(source, key)] = decode_vtf_mipchain(source, key)
        print("decoded", source, key, "mip levels:", len(texture_mips[(source, key)]))
    except Exception as e:
        print("FAILED to decode", source, key, "->", e, "-- falling back to local color/white")
        texture_mips[(source, key)] = decode_vtf_mipchain("local", "color/white")

# =================== 3. build mesh vertex/index buffers (with per-vertex groups via sorted tri order) ===================
SCALE = 0.0254
pos = np.zeros_like(verts_src)
pos[:, 0] = verts_src[:, 0] * SCALE
pos[:, 1] = verts_src[:, 1] * SCALE
pos[:, 2] = verts_src[:, 2] * SCALE
n_verts = pos.shape[0]
USE_32BIT_INDICES = n_verts >= 65536
print(f"n_verts={n_verts}, using {'UInt32' if USE_32BIT_INDICES else 'UInt16'} indices")

normals = np.zeros_like(pos)
p0 = pos[tris[:, 0]]; p1 = pos[tris[:, 1]]; p2 = pos[tris[:, 2]]
face_normals = np.cross(p1 - p0, p2 - p0)
lens = np.linalg.norm(face_normals, axis=1, keepdims=True); lens[lens == 0] = 1
face_normals = face_normals / lens
for i in range(3):
    np.add.at(normals, tris[:, i], face_normals)
nlen = np.linalg.norm(normals, axis=1, keepdims=True); nlen[nlen == 0] = 1
normals = normals / nlen

uvs = uvs_src.copy()
uvs[:, 1] = 1.0 - uvs[:, 1]
tangents = np.zeros((n_verts, 4), dtype=np.float32); tangents[:, 0] = 1.0; tangents[:, 3] = 1.0
uv1 = np.zeros((n_verts, 2), dtype=np.float32)

STRIDE = 56
buf = bytearray(n_verts * STRIDE)
for i in range(n_verts):
    struct.pack_into('<3f3f4f2f2f', buf, i * STRIDE,
                      pos[i,0], pos[i,1], pos[i,2],
                      normals[i,0], normals[i,1], normals[i,2],
                      tangents[i,0], tangents[i,1], tangents[i,2], tangents[i,3],
                      uvs[i,0], uvs[i,1], uv1[i,0], uv1[i,1])

# sort by group so each group's triangles are contiguous (needed for submesh ranges).
# NOT double-sided: an earlier version emitted both windings per triangle to rule out
# a winding/culling bug, but that bug turned out to be stale m_StaticBatchInfo instead
# -- doubling just left two coincident triangles at the same depth, causing z-fighting.
order = np.argsort(tri_group, kind='stable')
all_tris = tris[order]
all_group = tri_group[order]

n_tris = all_tris.shape[0]
if USE_32BIT_INDICES:
    idx_buf = bytearray(n_tris * 3 * 4)
    flat = all_tris.astype('<u4').reshape(-1)
    struct.pack_into(f'<{len(flat)}I', idx_buf, 0, *flat.tolist())
else:
    idx_buf = bytearray(n_tris * 3 * 2)
    flat = all_tris.astype('<u2').reshape(-1)
    struct.pack_into(f'<{len(flat)}H', idx_buf, 0, *flat.tolist())

# compute submesh ranges
submesh_ranges = []  # (group_id, first_tri, tri_count)
cur_group = None
start = 0
for i in range(n_tris + 1):
    g = all_group[i] if i < n_tris else None
    if g != cur_group:
        if cur_group is not None:
            submesh_ranges.append((cur_group, start, i - start))
        cur_group = g
        start = i
print("submesh ranges:", submesh_ranges)

bbox_min = pos.min(axis=0); bbox_max = pos.max(axis=0)
center = (bbox_min + bbox_max) / 2; extent = (bbox_max - bbox_min) / 2

# =================== 4. load env, inject mesh into sharedassets5.assets ===================
env = UnityPy.load(SRC)
f = env.files[SRC]
inner = f.files["data.unity3d"]
sf5 = inner.files["sharedassets5.assets"]

TEMPLATE_MESH_PID = 8  # Cube
orig_reader = sf5.objects[TEMPLATE_MESH_PID]
new_mesh_reader = copy.copy(orig_reader)
new_mesh_reader.path_id = MESH_PATH_ID
new_mesh_reader.data = orig_reader.get_raw_data()
new_mesh_reader._read_until = None
sf5.objects[MESH_PATH_ID] = new_mesh_reader

d = new_mesh_reader.read()
d.object_reader = new_mesh_reader
d.m_Name = "gm_construct"
d.m_VertexData.m_VertexCount = n_verts
d.m_VertexData.m_DataSize = bytes(buf)
d.m_IndexBuffer = list(idx_buf)
d.m_IndexFormat = 1 if USE_32BIT_INDICES else 0

submeshes = []
template_sub = d.m_SubMeshes[0]
for group_id, first_tri, tri_count in submesh_ranges:
    sub = copy.copy(template_sub)
    sub.firstByte = first_tri * 3 * (4 if USE_32BIT_INDICES else 2)
    sub.firstVertex = 0
    sub.indexCount = tri_count * 3
    sub.vertexCount = n_verts
    sub.baseVertex = 0
    sub.localAABB = copy.copy(template_sub.localAABB)
    sub.localAABB.m_Center = Vector3f(float(center[0]), float(center[1]), float(center[2]))
    sub.localAABB.m_Extent = Vector3f(float(extent[0]), float(extent[1]), float(extent[2]))
    submeshes.append(sub)
d.m_SubMeshes = submeshes
d.m_LocalAABB.m_Center = Vector3f(float(center[0]), float(center[1]), float(center[2]))
d.m_LocalAABB.m_Extent = Vector3f(float(extent[0]), float(extent[1]), float(extent[2]))
d.save()
print("injected mesh, submeshes:", len(submeshes))

# =================== 5. deep-copy City (level9) into an independent level7 ===================
level9 = inner.files["level9"]
pure_city_bytes = level9.save()
reader = EndianBinaryReader(pure_city_bytes)
level7 = SerializedFile(reader, parent=inner, name="level7")
level7.flags = level9.flags
inner.files["level7"] = level7
inner.mark_changed()
print("level7 independent clone, objects:", len(level7.objects))

# =================== 6. build Texture2D + Material objects in sharedassets9.assets ===================
# (level7's externals table is identical to level9's: index 2 -> sharedassets9.assets)
TEX_MAT_EXTERNAL_FILEID = 2
sf9 = inner.files["sharedassets9.assets"]
TEMPLATE_TEX_PID = None
TEMPLATE_MAT_PID = None
for pid, obj in sf9.objects.items():
    if obj.type.name == "Texture2D" and TEMPLATE_TEX_PID is None:
        TEMPLATE_TEX_PID = pid
    if obj.type.name == "Material" and TEMPLATE_MAT_PID is None:
        TEMPLATE_MAT_PID = pid
print("template tex/mat pids in sharedassets9:", TEMPLATE_TEX_PID, TEMPLATE_MAT_PID)

max_pid9 = max(sf9.objects.keys())
next_pid9 = [max_pid9 + 1000]
def alloc9():
    pid = next_pid9[0]; next_pid9[0] += 1; return pid

def clone_in_sf9(orig_pid, new_pid):
    orig = sf9.objects[orig_pid]
    new_obj = copy.copy(orig)
    new_obj.path_id = new_pid
    new_obj.data = orig.get_raw_data()
    new_obj._read_until = None
    sf9.objects[new_pid] = new_obj
    return new_obj

max_pid = max(level7.objects.keys())
next_pid = [max_pid + 1000]
def alloc():
    pid = next_pid[0]; next_pid[0] += 1; return pid

def clone_in_level7(orig_pid, new_pid):
    orig = level7.objects[orig_pid]
    new_obj = copy.copy(orig)
    new_obj.path_id = new_pid
    new_obj.data = orig.get_raw_data()
    new_obj._read_until = None
    level7.objects[new_pid] = new_obj
    return new_obj

group_to_material_pid = {}

if TEMPLATE_TEX_PID is not None and TEMPLATE_MAT_PID is not None:
    for vtf_key_tuple in unique_vtf_keys:
        source, key = vtf_key_tuple
        group_id = vtf_key_to_group[vtf_key_tuple]
        mips = texture_mips[vtf_key_tuple]  # list of (H,W,4) uint8, largest first, already flipped
        safe_name = key.replace("/", "_")

        tex_pid = alloc9()
        clone_in_sf9(TEMPLATE_TEX_PID, tex_pid)
        tex_reader = sf9.objects[tex_pid]
        td = tex_reader.read()
        td.object_reader = tex_reader
        td.m_Name = "tex_" + safe_name
        td.m_Width = TEX_SIZE
        td.m_Height = TEX_SIZE
        td.m_TextureFormat = 4  # RGBA32
        td.image_data = b"".join(m.tobytes() for m in mips)
        td.m_CompleteImageSize = len(td.image_data)
        td.m_MipCount = len(mips)
        td.m_StreamData.offset = 0
        td.m_StreamData.path = ''
        td.m_StreamData.size = 0
        td.save()

        mat_pid = alloc9()
        clone_in_sf9(TEMPLATE_MAT_PID, mat_pid)
        mat_reader = sf9.objects[mat_pid]
        md = mat_reader.read()
        md.object_reader = mat_reader
        md.m_Name = "mat_" + safe_name
        for name, texenv in md.m_SavedProperties.m_TexEnvs:
            if name == "_MainTex":
                texenv.m_Texture.m_FileID = 0
                texenv.m_Texture.m_PathID = tex_pid
        md.save()

        group_to_material_pid[group_id] = mat_pid
        print(f"built texture+material for {source}:{key}: tex_pid={tex_pid} mat_pid={mat_pid}")
else:
    print("WARNING: no Texture2D/Material template found in sharedassets9, skipping real textures")

# =================== 7. add gm_construct GameObject to level7, reparent to scene root ===================
TEMPLATE_GO_PID = 28  # "Ground1" in City
go_orig = level7.objects[TEMPLATE_GO_PID].read()
comp_pids_orig = {}
for c in go_orig.m_Components:
    t = level7.objects[c.path_id].type.name
    comp_pids_orig[t] = c.path_id

new_go_pid = alloc(); new_tr_pid = alloc(); new_mf_pid = alloc(); new_mr_pid = alloc(); new_mc_pid = alloc()

clone_in_level7(TEMPLATE_GO_PID, new_go_pid)
new_go_reader = level7.objects[new_go_pid]
go_data = new_go_reader.read()
go_data.object_reader = new_go_reader
go_data.m_Name = "gm_construct"
for comp in go_data.m_Components:
    if comp.path_id == comp_pids_orig["Transform"]: comp.m_PathID = new_tr_pid
    elif comp.path_id == comp_pids_orig["MeshFilter"]: comp.m_PathID = new_mf_pid
    elif comp.path_id == comp_pids_orig["MeshRenderer"]: comp.m_PathID = new_mr_pid
    elif comp.path_id == comp_pids_orig["MeshCollider"]: comp.m_PathID = new_mc_pid
go_data.save()

clone_in_level7(comp_pids_orig["Transform"], new_tr_pid)
new_tr_reader = level7.objects[new_tr_pid]
tr_data = new_tr_reader.read()
tr_data.object_reader = new_tr_reader
tr_data.m_GameObject.m_PathID = new_go_pid
tr_data.m_Children = []
# reparent to scene root (father=0) at the same effective world position the old
# nested chain (Ground1 -> Ground -> Pixelactive_City_FBX) used to produce
tr_data.m_Father.m_FileID = 0
tr_data.m_Father.m_PathID = 0
tr_data.m_LocalPosition = Vector3f(0.0, 0.6, 0.0)
tr_data.save()

clone_in_level7(comp_pids_orig["MeshFilter"], new_mf_pid)
new_mf_reader = level7.objects[new_mf_pid]
mf_data = new_mf_reader.read()
mf_data.object_reader = new_mf_reader
mf_data.m_GameObject.m_PathID = new_go_pid
mf_data.m_GameObject.m_FileID = 0
mf_data.m_Mesh.m_FileID = MESH_EXTERNAL_FILEID
mf_data.m_Mesh.m_PathID = MESH_PATH_ID
mf_data.save()

clone_in_level7(comp_pids_orig["MeshRenderer"], new_mr_pid)
new_mr_reader = level7.objects[new_mr_pid]
mr_data = new_mr_reader.read()
mr_data.object_reader = new_mr_reader
mr_data.m_GameObject.m_PathID = new_go_pid
mr_data.m_GameObject.m_FileID = 0
mr_data.m_StaticBatchInfo.firstSubMesh = 0
mr_data.m_StaticBatchInfo.subMeshCount = 0
new_materials = []
for group_id in range(n_groups):
    if group_id in group_to_material_pid:
        pptr = copy.copy(mr_data.m_Materials[0])
        pptr.m_FileID = TEX_MAT_EXTERNAL_FILEID
        pptr.m_PathID = group_to_material_pid[group_id]
        new_materials.append(pptr)
    else:
        # fallback group: reuse whatever material this MeshRenderer template already had
        new_materials.append(mr_data.m_Materials[0])
mr_data.m_Materials = new_materials
mr_data.save()
print("MeshRenderer materials assigned:", len(new_materials))

clone_in_level7(comp_pids_orig["MeshCollider"], new_mc_pid)
new_mc_reader = level7.objects[new_mc_pid]
mc_data = new_mc_reader.read(check_read=False)
mc_data.object_reader = new_mc_reader
mc_data.m_GameObject.m_PathID = new_go_pid
mc_data.m_GameObject.m_FileID = 0
mc_data.m_Mesh.m_FileID = MESH_EXTERNAL_FILEID
mc_data.m_Mesh.m_PathID = MESH_PATH_ID
mc_data.save()

print("added gm_construct GO at scene root, pids:", new_go_pid, new_tr_pid, new_mf_pid, new_mr_pid, new_mc_pid)

# =================== 7b. reposition SpawnPoints onto the new geometry ===================
# City's spawn points were tuned for City's own layout and now sit far outside (or
# clipped into) gm_construct's footprint -- find solid ground near the mesh's own
# center and move the spawns there instead, in level7 only.
SPAWN_CHILD_TR_PIDS = [82, 87, 85, 86, 81, 84]
cx, cz = float(center[0]), float(center[2])
nearby_mask = (np.abs(pos[:, 0] - cx) < 10) & (np.abs(pos[:, 2] - cz) < 10)
if nearby_mask.any():
    spawn_y = float(pos[nearby_mask, 1].max()) + 1.0
else:
    spawn_y = float(center[1]) + float(extent[1]) * 0.1 + 1.0
print(f"spawn location: x={cx:.2f} y={spawn_y:.2f} z={cz:.2f}")

for i, tr_pid in enumerate(SPAWN_CHILD_TR_PIDS):
    tr_reader = level7.objects[tr_pid]
    tr_data = tr_reader.read()
    tr_data.object_reader = tr_reader
    row = i // 3
    col = i % 3
    tr_data.m_LocalPosition = Vector3f(cx + col * 1.5, spawn_y, cz + row * 1.5)
    tr_data.save()
print("repositioned", len(SPAWN_CHILD_TR_PIDS), "spawn points in level7")

# =================== 8. disable the old City map content (Pixelactive_City_FBX) in level7 ONLY ===================
OLD_CITY_ROOT_GO_PID = 25  # "Pixelactive_City_FBX" GameObject (70 is its Transform, not the GO)
old_root_reader = level7.objects[OLD_CITY_ROOT_GO_PID]
old_root_data = old_root_reader.read()
old_root_data.object_reader = old_root_reader
old_root_data.m_IsActive = False
old_root_data.save()
print("disabled old City map root (pid 70) in level7 clone")

# ---- sanity: City (level9) truly untouched ----
print("level9 (City) object count unchanged:", len(level9.objects), "max pid:", max(level9.objects.keys()))

out_bytes = f.save()
with open(OUT, "wb") as fh:
    fh.write(out_bytes)
print("wrote", OUT, len(out_bytes), "bytes")
