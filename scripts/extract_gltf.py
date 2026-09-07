import struct
import numpy as np
from pygltflib import GLTF2

GLTF_DIR = "/home/cathead/Downloads/hypper/gm_construct_gltf_extracted"
OUT_NPZ = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/gm_construct_gltf_mesh.npz"

g = GLTF2().load(f"{GLTF_DIR}/untitled.gltf")
with open(f"{GLTF_DIR}/untitled.bin", "rb") as f:
    blob = f.read()

def get_accessor_data(gltf, accessor_idx, blob):
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    fmt = {5126: 'f', 5125: 'I', 5123: 'H', 5121: 'B'}[acc.componentType]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count
    data = struct.unpack_from(f'<{count*n_comp}{fmt}', blob, start)
    return np.array(data, dtype=np.float64 if fmt == 'f' else np.int64).reshape(count, n_comp)

def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])

def node_local_matrix(node):
    if node.matrix is not None:
        m = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # glTF matrices are column-major
        return m
    t = np.array(node.translation or [0, 0, 0], dtype=np.float64)
    r = node.rotation or [0, 0, 0, 1]
    s = np.array(node.scale or [1, 1, 1], dtype=np.float64)
    rot = quat_to_matrix(r)
    m = np.eye(4)
    m[:3, :3] = rot * s[np.newaxis, :]  # scale columns
    m[:3, 3] = t
    return m

# build world transforms via scene graph traversal (parent -> children)
n_nodes = len(g.nodes)
world_mat = [None] * n_nodes
children_of = {i: [] for i in range(n_nodes)}
is_child = set()
for i, node in enumerate(g.nodes):
    for c in (node.children or []):
        children_of[i].append(c)
        is_child.add(c)
roots = [i for i in range(n_nodes) if i not in is_child]

def assign_world(i, parent_world):
    local = node_local_matrix(g.nodes[i])
    world = parent_world @ local
    world_mat[i] = world
    for c in children_of[i]:
        assign_world(c, world)

for r in roots:
    assign_world(r, np.eye(4))

print(f"nodes: {n_nodes}, roots: {len(roots)}")

verts = []
uvs = []
tris = []
tri_mat_id = []
tri_src = []

mat_names_gltf = [m.name for m in g.materials]
# normalize to match old pipeline's key format: strip "materials/" prefix, lowercase
def normalize_mat_name(name):
    if name is None:
        return "unknown"
    n = name.lower()
    if n.startswith("materials/"):
        n = n[len("materials/"):]
    return n

unique_mats = sorted(set(normalize_mat_name(m) for m in mat_names_gltf))
mat_to_id = {m: i for i, m in enumerate(unique_mats)}

skipped_nodraw = 0
processed_prims = 0

for ni, node in enumerate(g.nodes):
    if node.mesh is None:
        continue
    wm = world_mat[ni]
    mesh = g.meshes[node.mesh]
    for pi, prim in enumerate(mesh.primitives):
        if prim.attributes.POSITION is None or prim.indices is None:
            continue
        mat_idx = prim.material
        mat_name_raw = mat_names_gltf[mat_idx] if mat_idx is not None else None
        norm_name = normalize_mat_name(mat_name_raw)
        if "nodraw" in norm_name or "trigger" in norm_name or "skip" in norm_name or "hint" in norm_name:
            skipped_nodraw += 1
            continue

        pos = get_accessor_data(g, prim.attributes.POSITION, blob).astype(np.float64)
        pos_h = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        world_pos = (wm @ pos_h.T).T[:, :3]

        uv = get_accessor_data(g, prim.attributes.TEXCOORD_0, blob) if prim.attributes.TEXCOORD_0 is not None else np.zeros((len(pos), 2))

        idx = get_accessor_data(g, prim.indices, blob).reshape(-1)

        # glTF is right-handed; Unity is left-handed. Standard conversion (matching
        # Khronos' own UnityGLTF importer): negate X, and reverse triangle winding
        # to compensate for the handedness flip.
        world_pos[:, 0] *= -1

        base_idx = len(verts)
        verts.extend(world_pos.tolist())
        uvs.extend(uv.tolist())

        node_name = node.name or f"node_{ni}"
        mid = mat_to_id[norm_name]
        for t in range(0, len(idx), 3):
            a, b, c = int(idx[t]), int(idx[t+1]), int(idx[t+2])
            tris.append((base_idx + a, base_idx + c, base_idx + b))  # reversed winding
            tri_mat_id.append(mid)
            tri_src.append(f"{node_name}_prim{pi}")
        processed_prims += 1

print(f"processed {processed_prims} primitives, skipped {skipped_nodraw} (nodraw/trigger/etc)")
print(f"total verts: {len(verts)}  total tris: {len(tris)}")
print(f"unique materials: {len(unique_mats)}")

verts_arr = np.array(verts, dtype=np.float32)
uvs_arr = np.array(uvs, dtype=np.float32)
tris_arr = np.array(tris, dtype=np.uint32)
tri_mat_id_arr = np.array(tri_mat_id, dtype=np.int32)
tri_src_arr = np.array(tri_src)

print("bbox:", verts_arr.min(0), verts_arr.max(0))

np.savez(OUT_NPZ, verts=verts_arr, uvs=uvs_arr, tris=tris_arr, tri_mat_id=tri_mat_id_arr,
         mat_names=np.array(unique_mats), tri_src=tri_src_arr)
print("wrote", OUT_NPZ)
