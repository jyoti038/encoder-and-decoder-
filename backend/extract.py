import numpy as np

from dwt import DWTProcessor
from dct import DCTProcessor
from svd import SVDProcessor


class Extractor:

    def __init__(self, delta=16):

        self.delta = delta

        self.dwt = DWTProcessor()

        self.dct = DCTProcessor()

        self.svd = SVDProcessor()

    # ---------------------------------------------
    # Robust QIM Decoder
    # ---------------------------------------------
    def decode_qim(self, value):

        remainder = value % self.delta

        if remainder < self.delta / 4:
            return 0

        elif remainder < (3 * self.delta) / 4:
            return 1

        else:
            return 0

    # ---------------------------------------------
    # Extract One Bit
    # ---------------------------------------------
    def extract_bit(self, sigma):

        return self.decode_qim(sigma[0])

    # ---------------------------------------------
    # Adaptive Delta
    # ---------------------------------------------
    def adaptive_delta(self, energy):

        if energy > 5000:
            return 20

        elif energy > 2000:
            return 16

        else:
            return 12

    # ---------------------------------------------
    # Decode One Block
    # ---------------------------------------------
    def extract_block(self, block, energy):

        dct_block = self.dct.forward(block)

        self.delta = self.adaptive_delta(energy)

        positions = self.dct.mid_positions()[:3]

        votes = []

        for x, y in positions:

            votes.append(
                self.decode_qim(
                    dct_block[x, y]
                )
            )

        return 1 if sum(votes) >= 2 else 0

    # ---------------------------------------------
    # Decode Multiple Blocks
    # ---------------------------------------------
    def extract_blocks(self, blocks, total_bits):

        bits = []

        for block in blocks[:total_bits]:

            energy = self.dwt.block_energy(
                block["block"]
            )

            bit = self.extract_block(
                block["block"],
                energy
            )

            bits.append(bit)

        print("Recovered Full Bits:", bits)
        print("Recovered First 64:", bits[:64])

        return bits

    # ---------------------------------------------
    # Extract From Image
    # ---------------------------------------------
    def extract(self, coeff, total_bits):

        hl = coeff["HL"]

        blocks = self.dwt.split_into_blocks(hl)

        selected = blocks

        available = len(selected)

        print(f"Requested Bits : {total_bits}")
        print(f"Available Blocks : {available}")

        total_bits = min(total_bits, available)

        total_bits = (total_bits // 8) * 8

        print(f"Extracting Bits : {total_bits}")

        bits = self.extract_blocks(
            selected,
            total_bits
        )

        return bits

    # ---------------------------------------------
    # Bit Accuracy
    # ---------------------------------------------
    def bit_accuracy(self, original, recovered):

        n = min(len(original), len(recovered))

        if n == 0:
            return 0

        correct = 0

        for a, b in zip(original[:n], recovered[:n]):

            if a == b:
                correct += 1

        return (correct / n) * 100

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------
    def statistics(self, bits):

        return {

            "recovered_bits": len(bits),

            "capacity_used": len(bits)

        }