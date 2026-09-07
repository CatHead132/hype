import struct

SRC = "/home/cathead/Downloads/hypper/decompiled/raw/global-metadata.dat"
OUT = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/global-metadata-patched.dat"
NEW_STRING = b"CityTest"

with open(SRC, "rb") as f:
    data = bytearray(f.read())

sig, version = struct.unpack_from('<II', data, 0)
assert sig == 0xFAB11BAF and version == 31

NUM_HEADER_PAIRS = 31
header = []
for i in range(NUM_HEADER_PAIRS):
    off, cnt = struct.unpack_from('<ii', data, 8 + i*8)
    header.append([off, cnt])

stringLiteralOffset, stringLiteralCount = header[0]
stringLiteralDataOffset, stringLiteralDataCount = header[1]

num_entries = stringLiteralCount // 8
new_index = num_entries
new_data_index = stringLiteralDataCount  # append at end of data blob (relative offset)

print(f"current entries: {num_entries}, new_index will be: {new_index}")
print(f"stringLiteralData current size: {stringLiteralDataCount}, new string will be at relative offset {new_data_index}")

# Step 1: insert new 8-byte entry {length, dataIndex} at end of stringLiteral table
# table currently spans [stringLiteralOffset, stringLiteralOffset+stringLiteralCount)
insert_pos_1 = stringLiteralOffset + stringLiteralCount
new_entry = struct.pack('<ii', len(NEW_STRING), new_data_index)
data[insert_pos_1:insert_pos_1] = new_entry
shift1 = len(new_entry)  # 8 bytes

# Step 2: insert new string bytes at end of stringLiteralData blob (account for shift1)
insert_pos_2 = stringLiteralDataOffset + shift1 + stringLiteralDataCount
data[insert_pos_2:insert_pos_2] = NEW_STRING
shift2 = len(NEW_STRING)

# Step 3: update header
header[0][1] += shift1  # stringLiteralCount += 8
header[1][0] += shift1  # stringLiteralDataOffset shifts by shift1
header[1][1] += shift2  # stringLiteralDataCount += len(new string)
for i in range(2, NUM_HEADER_PAIRS):
    header[i][0] += shift1 + shift2  # everything after both insertions shifts by both

for i, (off, cnt) in enumerate(header):
    struct.pack_into('<ii', data, 8 + i*8, off, cnt)

with open(OUT, "wb") as f:
    f.write(data)

print(f"wrote {OUT}, new size {len(data)} (was {8+NUM_HEADER_PAIRS*8} header, orig file was", end=" ")
with open(SRC, "rb") as f:
    print(len(f.read()), ")")

# sanity: reread and verify
with open(OUT, "rb") as f:
    verify = f.read()
off, cnt = struct.unpack_from('<ii', verify, 8)
doff, dcnt = struct.unpack_from('<ii', verify, 16)
length, dataIdx = struct.unpack_from('<ii', verify, off + new_index*8)
s = verify[doff+dataIdx: doff+dataIdx+length]
print(f"VERIFY: new_index={new_index} length={length} dataIdx={dataIdx} string={s}")
print("NEW STRING LITERAL INDEX (use this for the token):", new_index)
