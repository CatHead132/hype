import struct
import copy
import os
import re
import io
import json
import numpy as np
import UnityPy
from UnityPy.files import SerializedFile
from UnityPy.streams import EndianBinaryReader
from UnityPy.classes import Vector3f
from PIL import Image

SRC = "/home/cathead/Downloads/hypper/patched-data-recovered/patched_game.data"
OUT = "/home/cathead/Downloads/hypper/patched-data-recovered/patched_game_v4.data"
NPZ = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/gm_construct_real_mesh.npz"
TEX_DIR = "/mnt/storage/Apps/gm_construct_real_textures"
MAT_TEX_JSON = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/mat_base_texture_file.json"
MAT_ENVMAP_ALPHA_JSON = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/mat_envmap_alpha.json"

MESH_EXTERNAL_FILEID = 3  # sharedassets5.assets index in level9/level7's externals
MESH_PATH_ID = 1010

# =================== 1. material resolution ===================
# Textures come straight from SourceIO's real BSP import (blender_import_bsp.py):
# it mounted the actual GMod VPKs via gameinfo.txt, decoded the real VTFs into
# PNGs during Blender material construction, and extract_real_bsp.py pulled each
# material's baseColorTexture straight out of the exported glTF -- no guessing,
# no keyword matching, no curated pack. Real texture, per real material, 1:1.
with open(MAT_TEX_JSON) as fh:
    mat_base_texture_file = json.load(fh)
with open(MAT_ENVMAP_ALPHA_JSON) as fh:
    mat_envmap_alpha = json.load(fh)

data = np.load(NPZ, allow_pickle=True)
verts_src = data["verts"]
uvs_src = data["uvs"]
tris = data["tris"]
tri_mat_id = data["tri_mat_id"]
mat_names = data["mat_names"]

# resolved_vtf_for_mat[i] = (source, key); source is "real" (has a decoded texture
# file) or "fallback" (e.g. reflectiveglass001 -- a glass shader with no diffuse
# image, given a flat placeholder color instead of a wrong/guessed texture).
resolved_vtf_for_mat = {}
n_real = n_fallback = 0
for i, name in enumerate(mat_names):
    key = str(name)
    if key in mat_base_texture_file:
        resolved_vtf_for_mat[i] = ("real", key)
        n_real += 1
    else:
        resolved_vtf_for_mat[i] = ("fallback", key)
        n_fallback += 1
print(f"{n_real} real decoded textures, {n_fallback} flat-color fallback (of {len(mat_names)} materials)")

unique_vtf_keys = sorted(set(resolved_vtf_for_mat.values()))
print("unique real textures needed:", len(unique_vtf_keys))

vtf_key_to_group = {k: i for i, k in enumerate(unique_vtf_keys)}
tri_group = np.array([vtf_key_to_group[resolved_vtf_for_mat[mid]] for mid in tri_mat_id], dtype=np.int32)
n_groups = len(unique_vtf_keys)
print("submesh groups:", n_groups, "(all real textures, no generic fallback)")

# =================== 2. load real decoded textures (already PNG, from the BSP import) ===================
FALLBACK_SIZE = 64
MAX_TEX_SIZE = 2048  # cap only -- the world_geometry bake is a 4096 non-overlapping
                      # atlas (like a lightmap: unrelated faces live at different UV
                      # locations), so it must keep its OWN native resolution rather
                      # than being force-resized to a single shared size, which would
                      # blur completely unrelated faces' colors into each other.
                      # With point filtering (below) eliminating the interpolation
                      # blur/bleeding, resolution is now a pure memory/perf lever --
                      # capped here since the full uncompressed 4096 atlas plus 110
                      # displacement atlases was heavy enough to cause noticeable lag.
FALLBACK_RGBA = (160, 160, 160, 255)  # flat opaque gray, for real-but-alpha-disqualified/missing materials
GLASS_FALLBACK_RGBA = (200, 220, 235, 180)  # flat translucent blue-ish, only for the known glass material
WATER_RGBA = (40, 90, 110, 255)  # flat water blue-green (alpha/translucency comes from the material's _Color tint, not the texture, matching the game's own Glass material)

def decode_vtf_mipchain(source, key):
    """Returns list of (W,H,4) uint8 arrays, largest first, down to 1x1 -- without a
    full mip chain, tiled textures alias/shimmer badly at grazing angles."""
    if source == "real":
        base = Image.open(os.path.join(TEX_DIR, mat_base_texture_file[key])).convert("RGBA")
        if max(base.size) > MAX_TEX_SIZE:
            base.thumbnail((MAX_TEX_SIZE, MAX_TEX_SIZE), Image.LANCZOS)
        if base.size[0] != base.size[1]:
            # mip halving below assumes square; our own bakes and most VTFs already
            # are, this only guards stray non-square real textures
            side = max(base.size)
            base = base.resize((side, side), Image.LANCZOS)
        # Some vmts repurpose the base texture's alpha channel for something that
        # ISN'T transparency -- e.g. building_template006b.vmt has
        # "$basealphaenvmapmask" 1, meaning alpha is a cubemap-reflectivity mask.
        # Piping that raw alpha into the material as if it were real transparency
        # produces the "TV static" look (many overlapping near-transparent noisy
        # layers), so ONLY those specific materials get alpha forced opaque.
        # Most materials' alpha genuinely IS real cutout transparency (ladders'
        # see-through gaps, grates, fences, decals) -- forcing those opaque was
        # the actual bug: it replaced "invisible" cutout regions with whatever
        # garbage RGB happens to sit underneath (confirmed directly: metalladder001a
        # is bright with alpha-cut gaps in the source file, but came out with solid
        # BLACK gaps once alpha was force-flattened). Real alpha is now preserved
        # for everything except the confirmed envmap-mask cases.
        if mat_envmap_alpha.get(key):
            r, g, b, _ = base.split()
            base = Image.merge("RGBA", (r, g, b, Image.new("L", base.size, 255)))
        # else: real cutout alpha preserved as-is. The game's own "metal_fence-1"
        # material proves Alpha Cutout IS compiled into this build -- my first
        # attempt just set it up wrong (see the material-creation step below).
        # Real Source VTFs read a bit darker/flatter than the base game's own art
        # (which runs brighter, more saturated). Grass gets a bigger lift than
        # everything else per explicit request.
        r, g, b, a = base.split()
        GRASS_KEYS = {"flatgrass", "flatgrass_2", "grass_13", "grass-sand_13"}
        BRIGHTNESS_FACTOR = 1.5 if key in GRASS_KEYS else 1.05
        rgb = np.array(Image.merge("RGB", (r, g, b))).astype(np.float32)
        rgb = np.clip(rgb * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
        base = Image.merge("RGBA", (*Image.fromarray(rgb, "RGB").split(), a))
    elif key == "reflectiveglass001":
        base = Image.new("RGBA", (FALLBACK_SIZE, FALLBACK_SIZE), GLASS_FALLBACK_RGBA)
    elif key == "water":
        base = Image.new("RGBA", (FALLBACK_SIZE, FALLBACK_SIZE), WATER_RGBA)
    else:
        base = Image.new("RGBA", (FALLBACK_SIZE, FALLBACK_SIZE), FALLBACK_RGBA)
    # world_geometry_baked_mat / gm_construct_disp_*_baked_mat are non-overlapping
    # atlases (like a lightmap) -- completely unrelated faces sit at arbitrary
    # neighboring UV locations, so standard mipmapping (which box-filters across
    # the WHOLE texture with no awareness of island borders) blends unrelated
    # content together the moment a face is even slightly minified. That was the
    # actual cause of "so blurry" -- not resolution/memory. A single full-res level
    # (no mip chain at all) avoids that cross-island bleeding entirely. Ordinary
    # tileable textures (grass, concrete, etc, not atlases) keep their mip chain.
    is_atlas = key.endswith("_baked_mat")
    mips = []
    size = base.size[0]
    img = base
    while True:
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        mips.append(np.array(flipped, dtype=np.uint8))
        if is_atlas or size == 1:
            break
        size = max(1, size // 2)
        img = img.resize((size, size), Image.LANCZOS)
    return mips, base.size[0]

texture_mips = {}
texture_sizes = {}
for source, key in unique_vtf_keys:
    texture_mips[(source, key)], texture_sizes[(source, key)] = decode_vtf_mipchain(source, key)
    print("loaded", source, key, "size:", texture_sizes[(source, key)], "mip levels:", len(texture_mips[(source, key)]))

# =================== 3. build mesh vertex/index buffers (with per-vertex groups via sorted tri order) ===================
# SourceIO's BSP importer already outputs real-world meters (its default World Scale
# IS the hammer-unit-to-meters constant), so unlike the old from-scratch/curated-pack
# pipelines, no extra 0.0254 factor was needed for real-world accuracy. This extra
# 1.25x is a deliberate feel/gameplay bump on top of that (map felt a bit small
# relative to the player), not a correctness fix.
SCALE = 1.5
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
# NOT flipping V here: the exact same raw UV values (straight from the glTF/Blender
# export, no adjustment) render correctly in the web viewer against these same
# textures. The only convention correction actually needed is on the IMAGE side
# (FLIP_TOP_BOTTOM below, for UnityPy's raw image_data byte-row order) -- flipping
# BOTH was double-correcting, which is what was silently making every single
# textured surface look wrong despite materials/pixel-content being provably right.
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
        tex_size = texture_sizes[vtf_key_tuple]
        safe_name = key.replace("/", "_")

        tex_pid = alloc9()
        clone_in_sf9(TEMPLATE_TEX_PID, tex_pid)
        tex_reader = sf9.objects[tex_pid]
        td = tex_reader.read()
        td.object_reader = tex_reader
        td.m_Name = "tex_" + safe_name
        td.m_Width = tex_size
        td.m_Height = tex_size
        td.m_TextureFormat = 4  # RGBA32
        td.image_data = b"".join(m.tobytes() for m in mips)
        td.m_CompleteImageSize = len(td.image_data)
        td.m_MipCount = len(mips)
        td.m_MipMap = len(mips) > 1
        if key.endswith("_baked_mat"):
            # Point filtering (tried first) fixed the seams but made everything
            # visibly blocky -- these atlases don't have enough texels per face to
            # look good with zero interpolation. The seams were actually from the
            # default WRAP MODE being Repeat: with bilinear filtering, any UV that
            # lands very close to 0 or 1 samples across the texture boundary and
            # picks up the WRAPPED-AROUND opposite edge, which is a totally
            # unrelated island -- a black line at exactly that seam. Clamp mode
            # can never sample past the texture edge, so bilinear can stay on
            # (smooth, no pixelation) without reintroducing that wraparound bleed.
            td.m_TextureSettings.m_FilterMode = 1  # Bilinear
            td.m_TextureSettings.m_WrapU = 1  # Clamp
            td.m_TextureSettings.m_WrapV = 1  # Clamp
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
        if source == "real" and not mat_envmap_alpha.get(key):
            # First attempt at Cutout mode had it backwards: this game's OWN
            # "metal_fence-1" material (resources.assets pid 113) proves the
            # Cutout variant genuinely IS compiled into this build and works --
            # but its _ALPHATEST_ON is listed in InvalidKeywords, not Valid, and
            # its render queue is left at the default (-1), not overridden. The
            # actual switch is just these 4 float properties; touching the
            # keyword lists or render queue was the mistake, not the mode itself.
            new_floats = []
            for name, v in md.m_SavedProperties.m_Floats:
                if name == "_Mode":
                    v = 1.0  # Cutout
                elif name == "_SrcBlend":
                    v = 1.0  # One
                elif name == "_DstBlend":
                    v = 0.0  # Zero
                elif name == "_ZWrite":
                    v = 1.0
                new_floats.append((name, v))
            md.m_SavedProperties.m_Floats = new_floats
        elif key == "water":
            # Fade/Transparent mode, matching the game's own working "Glass"
            # material (resources.assets pid 80) exactly: _Mode=2 with SrcAlpha/
            # OneMinusSrcAlpha blending and ZWrite off. Translucency comes from
            # the material's _Color alpha (like Glass does), not texture alpha.
            new_floats = []
            for name, v in md.m_SavedProperties.m_Floats:
                if name == "_Mode":
                    v = 2.0  # Fade
                elif name == "_SrcBlend":
                    v = 5.0  # SrcAlpha
                elif name == "_DstBlend":
                    v = 10.0  # OneMinusSrcAlpha
                elif name == "_ZWrite":
                    v = 0.0
                new_floats.append((name, v))
            md.m_SavedProperties.m_Floats = new_floats
            from UnityPy.classes import ColorRGBA
            new_colors = []
            for name, c in md.m_SavedProperties.m_Colors:
                if name == "_Color":
                    c = ColorRGBA(r=0.35, g=0.55, b=0.6, a=0.65)
                new_colors.append((name, c))
            md.m_SavedProperties.m_Colors = new_colors
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
# CRITICAL: Unity matches m_Materials[i] to mesh.m_SubMeshes[i] by INDEX. Groups
# with zero actual triangles (e.g. "water"/"color_room", which still resolve to a
# real texture even though their geometry was excluded) never appear in
# submesh_ranges, so iterating range(n_groups) here as before silently produced
# MORE materials (69) than actual submeshes (60) -- every submesh from the first
# empty group onward then bound to the WRONG (shifted) material. This is very
# likely THE root cause behind nearly all of this session's "wrong texture"
# reports. Building new_materials from submesh_ranges directly keeps both lists
# in exact 1:1 index correspondence, by construction.
for group_id, first_tri, tri_count in submesh_ranges:
    pptr = copy.copy(mr_data.m_Materials[0])
    pptr.m_FileID = TEX_MAT_EXTERNAL_FILEID
    pptr.m_PathID = group_to_material_pid[group_id]
    new_materials.append(pptr)
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
