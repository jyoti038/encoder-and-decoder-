from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from embed import Embedder

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

y = data["y_channel"]

dwt = DWTProcessor()

coeff = dwt.forward(y)

blocks = dwt.split_into_blocks(coeff["HL"])

ranked = dwt.rank_blocks(blocks)

selected = dwt.select_blocks(ranked, 0.30)

embed = Embedder()

bits = [1,0,1,1,0,1,0,0,1,1]

embedded = embed.embed_message(selected, bits)

print(embed.statistics(bits, selected))

print()

print("Modified Blocks :", len(embedded))

encoded = embed.embed(
    coeff,
    bits
)

print()

print("Encoded Image Shape")

print(encoded.shape)