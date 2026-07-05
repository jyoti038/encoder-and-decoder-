from preprocess import ImagePreprocessor
from dwt import DWTProcessor

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

y = data["y_channel"]

dwt = DWTProcessor(
    wavelet="haar",
    level=1
)

# Forward DWT
coeff = dwt.forward(y)

print("Bands:")
print("LL :", coeff["LL"].shape)
print("LH :", coeff["LH"].shape)
print("HL :", coeff["HL"].shape)
print("HH :", coeff["HH"].shape)

print("\nStatistics")
print(dwt.statistics(coeff))

print("\nEnergy")
print(dwt.energy_report(coeff))

print("\nBest Embedding Band")
print(dwt.best_embedding_band(coeff))

print("\nCapacity")
print(dwt.calculate_capacity(coeff, "HL"))

# Reconstruction
reconstructed = dwt.inverse(coeff)

print("\nReconstructed Shape")
print(reconstructed.shape)

print("\n----------------------------")
print("Block Analysis")
print("----------------------------")

blocks = dwt.split_into_blocks(
    coeff["HL"]
)

print("Total Blocks:", len(blocks))

ranked = dwt.rank_blocks(
    blocks
)

selected = dwt.select_blocks(
    ranked,
    percentage=0.30
)

print("Selected Blocks:", len(selected))

print("HL Entropy:", dwt.calculate_entropy(coeff["HL"]))

print("Highest Block Energy:", ranked[0]["energy"])

print("Lowest Block Energy:", ranked[-1]["energy"])