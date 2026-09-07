import os
import re
import io
import json
import numpy as np
from srctools.bsp import BSP
from srctools.vpk import VPK
from srctools.vtf import VTF

NPZ = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/gm_construct_gltf_mesh.npz"
GM_DIR = "/home/cathead/Downloads/hypper/gm_construct"
GMOD_DIR = "/home/cathead/Downloads/GarrysMod"
OUT_DIR = "/home/cathead/Downloads/hypper/gm_construct_viewer"
os.makedirs(OUT_DIR, exist_ok=True)
TEX_DIR = os.path.join(OUT_DIR, "textures")
os.makedirs(TEX_DIR, exist_ok=True)

# ---- same resolution cascade as build_final_v3.py ----
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
    return "materials\\" + vtf_key.replace("/", "\\") + ".vtf"

bsp = BSP("/home/cathead/Downloads/GarrysMod/garrysmod/maps/gm_construct.bsp")
bsp_pak = bsp.pakfile
bsp_pak_names_lower = {n.lower(): n for n in bsp_pak.namelist()}

GMOD_VPK_PATHS = [
    os.path.join(GMOD_DIR, "sourceengine", "hl2_textures_dir.vpk"),
    os.path.join(GMOD_DIR, "sourceengine", "content_hl2_dir.vpk"),
    os.path.join(GMOD_DIR, "sourceengine", "content_cstrike_dir.vpk"),
]
gmod_vpks = [VPK(p) for p in GMOD_VPK_PATHS if os.path.exists(p)]
gmod_vtf_source = {}
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

def read_gmod_vmt(key):
    if key not in gmod_vmt_source:
        return None
    vi, path = gmod_vmt_source[key]
    return gmod_vpks[vi][path].read().decode(errors="ignore")

def resolve_gmod_key(mat_name, depth=0):
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

data = np.load(NPZ, allow_pickle=True)
verts = data["verts"]
uvs = data["uvs"]
tris = data["tris"]
tri_mat_id = data["tri_mat_id"]
mat_names = data["mat_names"]
tri_src = data["tri_src"] if "tri_src" in data else np.array(["unknown"] * len(tris))

resolved = {}
resolution_kind = {}  # i -> "exact_local" / "exact_gmod" / "closest"
for i, name in enumerate(mat_names):
    local = resolve_local_vtf_key(str(name))
    if local:
        resolved[i] = ("local", local)
        resolution_kind[i] = "exact"
        continue
    gmod = resolve_gmod_key(str(name))
    if gmod:
        resolved[i] = ("gmod", gmod)
        resolution_kind[i] = "exact"
        continue
    resolved[i] = ("local", closest_vtf_key(str(name)))
    resolution_kind[i] = "closest_guess"

unique_keys = sorted(set(resolved.values()))
print(f"{len(unique_keys)} unique textures to export")

def decode_and_save(source, key):
    safe = key.replace("/", "_")
    out_png = os.path.join(TEX_DIR, f"{safe}.png")
    if source == "local":
        path = os.path.join(GM_DIR, vtf_disk_filename(key))
        with open(path, "rb") as fh:
            raw = fh.read()
    else:
        vi, real_name = gmod_vtf_source[key]
        raw = gmod_vpks[vi][real_name].read()
    vtf = VTF.read(io.BytesIO(raw))
    frame = vtf.get()
    img = frame.to_PIL().convert("RGBA")
    img = img.resize((256, 256))
    img.save(out_png)
    return f"textures/{safe}.png"

key_to_pngpath = {}
for source, key in unique_keys:
    try:
        key_to_pngpath[(source, key)] = decode_and_save(source, key)
    except Exception as e:
        print("FAILED", source, key, e)
        key_to_pngpath[(source, key)] = None
print("textures exported:", len(key_to_pngpath))

# ---- write OBJ + MTL ----
MTL_PATH = os.path.join(OUT_DIR, "gm_construct.mtl")
with open(MTL_PATH, "w") as f:
    for i, name in enumerate(mat_names):
        source, key = resolved[i]
        png = key_to_pngpath.get((source, key))
        f.write(f"newmtl mat_{i}\n")
        f.write("Kd 1 1 1\n")
        if png:
            f.write(f"map_Kd {png}\n")
        f.write("\n")

OBJ_PATH = os.path.join(OUT_DIR, "gm_construct.obj")
SCALE = 0.0254
order = np.argsort(tri_mat_id, kind="stable")
sorted_tris = tris[order]
sorted_matid = tri_mat_id[order]
sorted_src = tri_src[order]

with open(OBJ_PATH, "w") as f:
    f.write("mtllib gm_construct.mtl\n")
    for v in verts:
        f.write(f"v {v[0]*SCALE:.6f} {v[1]*SCALE:.6f} {v[2]*SCALE:.6f}\n")
    for uv in uvs:
        f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
    current = None
    for tri, mid in zip(sorted_tris, sorted_matid):
        if mid != current:
            f.write(f"usemtl mat_{mid}\n")
            current = mid
        a, b, c = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

# viewer_data.json: per-triangle info IN THE SAME ORDER triangles appear in the OBJ
# (sorted by material, matching what three.js's OBJLoader will produce as faceIndex order)
viewer_data = {
    "triangles": [
        {"material": str(mat_names[mid]), "source": str(src), "resolved_texture": f"{resolved[mid][0]}:{resolved[mid][1]}", "kind": resolution_kind[mid]}
        for mid, src in zip(sorted_matid.tolist(), sorted_src.tolist())
    ],
    "materials": [str(n) for n in mat_names],
}
with open(os.path.join(OUT_DIR, "viewer_data.json"), "w") as f:
    json.dump(viewer_data, f)

print(f"\nwrote {OBJ_PATH}")
print(f"wrote {MTL_PATH}")
print(f"wrote viewer_data.json ({len(viewer_data['triangles'])} triangle entries)")
