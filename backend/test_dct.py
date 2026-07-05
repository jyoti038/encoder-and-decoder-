from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from dct import DCTProcessor

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

y = data["y_channel"]

dwt = DWTProcessor()

coeff = dwt.forward(y)

blocks = dwt.split_into_blocks(coeff["HL"])

ranked = dwt.rank_blocks(blocks)

selected = dwt.select_blocks(ranked, 0.30)

dct = DCTProcessor()

dct_blocks = dct.process_blocks(selected)

print("Selected Blocks :", len(dct_blocks))

print()

print("First Block Stats")

print(

    dct.statistics(

        dct_blocks[0]["block"]

    )

)

print()

print("Mid Frequency")

print(

    dct.get_mid_band(

        dct_blocks[0]["block"]

    )

)