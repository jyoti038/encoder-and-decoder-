from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from dct import DCTProcessor
from svd import SVDProcessor

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

y = data["y_channel"]

dwt = DWTProcessor()

coeff = dwt.forward(y)

blocks = dwt.split_into_blocks(
    coeff["HL"]
)

ranked = dwt.rank_blocks(blocks)

selected = dwt.select_blocks(
    ranked,
    0.30
)

dct = DCTProcessor()

dct_blocks = dct.process_blocks(
    selected
)

svd = SVDProcessor()

block = dct_blocks[0]["block"]

U, S, VT = svd.forward(block)

print()

print("Sigma")

print(S)

print()

print(svd.statistics(S))

print()

modified = svd.embed_bit(
    S,
    1
)

print(modified)

recovered = svd.inverse(
    U,
    modified,
    VT
)

print()

print(recovered.shape)