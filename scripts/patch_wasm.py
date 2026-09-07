import struct

SRC = "/home/cathead/Downloads/hypper/decompiled/raw/game.wasm"
OUT = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/game-patched.wasm"

NEW_STRING_INDEX = 15933
TYPE_TAG_STRING_LITERAL = 5
TOKEN = (TYPE_TAG_STRING_LITERAL << 29) | (NEW_STRING_INDEX << 1) | 1
print("TOKEN:", TOKEN, hex(TOKEN))

RESOLVER_CALL_ABS_IDX = 2305  # func[2305] absolute index (the "ensure resolved" wrapper)
SETMAP_ABS_IDX = 73192
CURRENTMAP_OFFSET = 28  # field offset within MapGroup object (0x1C)

def read_leb128(data, pos):
    result = 0; shift = 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7f) << shift
        if (b & 0x80) == 0: break
        shift += 7
    return result, pos

def write_uleb128(n):
    assert n >= 0, f"write_uleb128 called with negative n={n}"
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def write_sleb128(n):
    out = bytearray()
    more = True
    while more:
        byte = n & 0x7f
        n >>= 7
        if (n == 0 and (byte & 0x40) == 0) or (n == -1 and (byte & 0x40) != 0):
            more = False
        else:
            byte |= 0x80
        out.append(byte)
    return bytes(out)

import time
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

with open(SRC, "rb") as f:
    data = bytearray(f.read())
log("loaded file")

pos = 8
sections = []  # (id, header_pos, content_pos, content_size)
while pos < len(data):
    sec_id = data[pos]
    p2 = pos + 1
    sec_size, p2 = read_leb128(data, p2)
    content_pos = p2
    sections.append([sec_id, pos, content_pos, sec_size])
    pos = content_pos + sec_size

sec_by_id = {s[0]: s for s in sections}

# --- compute num_func_imports ---
import_sec = sec_by_id[2]
p = import_sec[2]
count, p = read_leb128(data, p)
num_func_imports = 0
for i in range(count):
    mod_len, p = read_leb128(data, p); p += mod_len
    name_len, p = read_leb128(data, p); p += name_len
    kind = data[p]; p += 1
    if kind == 0:
        num_func_imports += 1
        _, p = read_leb128(data, p)
    elif kind == 1:
        p += 1
        flags = data[p]; p += 1
        _, p = read_leb128(data, p)
        if flags & 1:
            _, p = read_leb128(data, p)
    elif kind == 2:
        flags = data[p]; p += 1
        _, p = read_leb128(data, p)
        if flags & 1:
            _, p = read_leb128(data, p)
    elif kind == 3:
        p += 2
    else:
        raise Exception("bad kind")
print("num_func_imports:", num_func_imports)

# ============ 1. Patch Memory section: initial pages 512 -> 513 ============
mem_sec = sec_by_id[5]
p = mem_sec[2]
mem_count, p = read_leb128(data, p)
assert mem_count == 1
flags = data[p]
flags_pos = p
p += 1
initial, p = read_leb128(data, p)
initial_pos_start = flags_pos + 1
initial_bytes_old = data[initial_pos_start:p]
NEW_SLOT_ADDR = initial * 65536  # start of the brand new page
print("old initial pages:", initial, "-> new slot address:", NEW_SLOT_ADDR)
new_initial = initial + 1
new_initial_bytes = write_uleb128(new_initial)
# splice (handle possible max field after, flags bit0 indicates max present)
has_max = flags & 1
if has_max:
    max_pos = p
    max_val, p_after_max = read_leb128(data, p)
    max_bytes_old = data[max_pos:p_after_max]
else:
    max_bytes_old = b""
    p_after_max = p

mem_content_old_end = p_after_max
new_mem_bytes = bytes([flags]) + new_initial_bytes + max_bytes_old
old_mem_bytes = data[flags_pos:mem_content_old_end]
delta_mem = len(new_mem_bytes) - len(old_mem_bytes)
data[flags_pos:mem_content_old_end] = new_mem_bytes
print("memory section byte delta:", delta_mem)

# adjust mem_sec size header + all subsequent section headers/positions (rebuild section list after each structural change is simplest)
log("before resync def")
def resync_sections():
    """Only walks far enough to find each of the known section IDs we care about
    (1..11), stopping each id at its FIRST occurrence. We must NOT keep walking past
    a section whose header/content-length is currently inconsistent (e.g. Code section
    after we've grown its content but before fixing its size header) -- doing so reads
    garbage and can coincidentally produce a bogus duplicate "section id" that would
    overwrite the correct entry. So: stop as soon as we've seen ids 1..11 once each."""
    global sections, sec_by_id, data
    pos = 8
    sections = []
    seen = set()
    wanted = set(range(1, 12))
    while pos < len(data):
        sec_id = data[pos]
        p2 = pos + 1
        sec_size, p2 = read_leb128(data, p2)
        content_pos = p2
        sections.append([sec_id, pos, content_pos, sec_size])
        seen.add(sec_id)
        pos = content_pos + sec_size
        if wanted <= seen:
            break
    sec_by_id = {}
    for s in sections:
        if s[0] not in sec_by_id:
            sec_by_id[s[0]] = s

def rewrite_section_size_by_delta(sec_id, delta):
    """Adjust a section's LEB128 size header by a known byte delta (avoids relying on
    other sections' positions being already-consistent, which caused an infinite loop
    when new_size went negative)."""
    resync_sections()
    sid, header_pos, content_pos, old_size = sec_by_id[sec_id]
    new_size = old_size + delta
    print(f"  [rewrite_section_size_by_delta] sec_id={sec_id} header_pos={header_pos} content_pos={content_pos} old_size={old_size} delta={delta} new_size={new_size}")
    assert new_size >= 0, f"new_size went negative for section {sec_id}: {new_size}"
    old_size_bytes_len = content_pos - (header_pos + 1)
    new_size_bytes = write_uleb128(new_size)
    print(f"  old_size_bytes_len={old_size_bytes_len} new_size_bytes={new_size_bytes.hex()} len={len(new_size_bytes)}")
    data[header_pos+1:header_pos+1+old_size_bytes_len] = new_size_bytes
    size_header_delta = len(new_size_bytes) - old_size_bytes_len
    if size_header_delta != 0:
        # the size header itself changed length -- extremely unlikely here since sizes stay small,
        # but guard anyway by padding/truncating is unsafe; just assert it doesn't happen.
        raise Exception(f"size header byte-length changed for section {sec_id}, unhandled")
    resync_sections()
    # verify
    sid2, hp2, cp2, sz2 = sec_by_id[sec_id]
    print(f"  VERIFY after resync: sec_id={sec_id} header_pos={hp2} content_pos={cp2} size={sz2}")

log("about to rewrite mem section size")
rewrite_section_size_by_delta(5, delta_mem)  # Memory section

# ============ 2. Add new Data segment at NEW_SLOT_ADDR with the 4-byte TOKEN ============
resync_sections()
data_sec = sec_by_id[11]
p = data_sec[2]
seg_count, count_pos_end = read_leb128(data, p)
# insert new segment right after the count varint, at the START of segments (simplest insertion point)
new_count_bytes = write_uleb128(seg_count + 1)
old_count_bytes = data[p:count_pos_end]

new_segment = bytearray()
new_segment += bytes([0x00])  # flags=0 (active, memory index 0 implicit)
new_segment += bytes([0x41])  # i32.const
new_segment += write_sleb128(NEW_SLOT_ADDR)
new_segment += bytes([0x0B])  # end
token_bytes = struct.pack('<I', TOKEN)
new_segment += write_uleb128(len(token_bytes))
new_segment += token_bytes

insertion_point = count_pos_end  # right after old count varint
data[p:count_pos_end] = new_count_bytes
shift_after_count = len(new_count_bytes) - len(old_count_bytes)
insertion_point += shift_after_count
data[insertion_point:insertion_point] = bytes(new_segment)

delta_data = shift_after_count + len(new_segment)
log("about to rewrite data section size")
rewrite_section_size_by_delta(11, delta_data)
print("added data segment: addr=", NEW_SLOT_ADDR, "token_bytes=", token_bytes.hex())

# ============ 3. Patch SetMap function body: insert new branch for value==7 ============
resync_sections()
code_sec = sec_by_id[10]
p = code_sec[2]
log("resynced for code section")
func_count, p = read_leb128(data, p)
bodies = []
for i in range(func_count):
    body_size, body_size_end = read_leb128(data, p)
    bodies.append((p, body_size_end, body_size))  # (size_leb_start, body_start, size)
    p = body_size_end + body_size

local_idx = SETMAP_ABS_IDX - num_func_imports
size_leb_start, body_start, body_size = bodies[local_idx]
body = data[body_start:body_start+body_size]
print("SetMap body size:", body_size)

# find the marker sequence for: local.get 1 ; i32.const 6 ; i32.le_u  == 20 01 41 06 4D
marker = bytes([0x20, 0x01, 0x41, 0x06, 0x4D])
idx = body.find(marker)
assert idx != -1, "marker not found!"
print("found existing value<=6 check at body offset", idx)

# build new bytecode:
# local.get 1 ; i32.const 7 ; i32.eq ; if
#   i32.const NEW_SLOT_ADDR ; call RESOLVER_CALL_ABS_IDX  (ensure resolved, drop result -- func already void)
#   local.get 0 ; i32.const NEW_SLOT_ADDR ; i32.load 2 0 ; i32.store 2 CURRENTMAP_OFFSET
# end
new_code = bytearray()
new_code += bytes([0x20, 0x01])              # local.get 1
new_code += bytes([0x41]) + write_sleb128(7) # i32.const 7
new_code += bytes([0x46])                    # i32.eq
new_code += bytes([0x04, 0x40])              # if (void)
new_code += bytes([0x41]) + write_sleb128(NEW_SLOT_ADDR)  # i32.const NEW_SLOT_ADDR
new_code += bytes([0x10]) + write_uleb128(RESOLVER_CALL_ABS_IDX)  # call 2305
new_code += bytes([0x20, 0x00])              # local.get 0 (this)
new_code += bytes([0x41]) + write_sleb128(NEW_SLOT_ADDR)  # i32.const NEW_SLOT_ADDR
new_code += bytes([0x28, 0x02, 0x00])        # i32.load align=2 offset=0
new_code += bytes([0x36, 0x02]) + write_uleb128(CURRENTMAP_OFFSET)  # i32.store align=2 offset=28
new_code += bytes([0x0B])                    # end (if)

new_body = body[:idx] + bytes(new_code) + body[idx:]
delta_body = len(new_body) - len(body)
print("new SetMap body size:", len(new_body), "delta:", delta_body)

new_size_leb = write_uleb128(len(new_body))
old_size_leb_len = body_start - size_leb_start

data[body_start:body_start+body_size] = new_body
data[size_leb_start:size_leb_start+old_size_leb_len] = new_size_leb

delta_code = (len(new_size_leb) - old_size_leb_len) + delta_body
rewrite_section_size_by_delta(10, delta_code)

with open(OUT, "wb") as f:
    f.write(data)
print("wrote", OUT, len(data), "bytes (orig", end=" ")
with open(SRC, "rb") as f:
    print(len(f.read()), ")")
