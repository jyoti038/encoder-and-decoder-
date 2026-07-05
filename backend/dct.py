import cv2
import numpy as np


class DCTProcessor:

    def __init__(self, block_size=8):

        self.block_size = block_size

    # --------------------------------------------
    # Validate
    # --------------------------------------------

    def validate(self, block):

        if block is None:
            raise ValueError("Block is None.")

        if block.shape != (
            self.block_size,
            self.block_size
        ):
            raise ValueError(
                f"Expected {self.block_size}x{self.block_size} block."
            )

    # --------------------------------------------
    # Forward DCT
    # --------------------------------------------

    def forward(self, block):

        self.validate(block)

        return cv2.dct(
            block.astype(np.float32)
        )

    # --------------------------------------------
    # Inverse DCT
    # --------------------------------------------

    def inverse(self, coeff):

        return cv2.idct(
            coeff.astype(np.float32)
        )

    # --------------------------------------------
    # Forward All Blocks
    # --------------------------------------------

    def process_blocks(self, blocks):

        transformed = []

        for item in blocks:

            transformed.append({

                "row": item["row"],

                "col": item["col"],

                "energy": item["energy"],

                "block": self.forward(
                    item["block"]
                )

            })

        return transformed

    # --------------------------------------------
    # Inverse All Blocks
    # --------------------------------------------

    def reconstruct_blocks(self, coeff_blocks):

        blocks = []

        for item in coeff_blocks:

            blocks.append({

                "row": item["row"],

                "col": item["col"],

                "energy": item["energy"],

                "block": self.inverse(
                    item["block"]
                )

            })

        return blocks
    
        # --------------------------------------------
    # Mid Frequency Coordinates
    # --------------------------------------------

    def mid_positions(self):

        return [

            (2,3),

            (3,2),

            (3,3),

            (2,4),

            (4,2),

            (4,3),

            (3,4)

        ]

    # --------------------------------------------
    # Read Mid Frequency
    # --------------------------------------------

    def get_mid_band(self, coeff):

        values=[]

        for x,y in self.mid_positions():

            values.append(

                coeff[x,y]

            )

        return values

    # --------------------------------------------
    # Replace Mid Frequency
    # --------------------------------------------

    def set_mid_band(

        self,

        coeff,

        values

    ):

        coeff=coeff.copy()

        for (x,y),v in zip(

            self.mid_positions(),

            values

        ):

            coeff[x,y]=v

        return coeff
    
        # --------------------------------------------
    # Zigzag Scan
    # --------------------------------------------

    def zigzag(self, block):

        indexorder = sorted(

            ((x, y) for x in range(8) for y in range(8)),

            key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1])

        )

        return [

            block[i,j]

            for i,j in indexorder

        ]

    # --------------------------------------------
    # Block Energy
    # --------------------------------------------

    def energy(self, coeff):

        return np.sum(

            np.square(coeff)

        )

    # --------------------------------------------
    # Statistics
    # --------------------------------------------

    def statistics(self, coeff):

        return {

            "min":float(np.min(coeff)),

            "max":float(np.max(coeff)),

            "mean":float(np.mean(coeff)),

            "std":float(np.std(coeff))

        }
    
    