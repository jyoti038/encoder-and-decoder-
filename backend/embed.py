import numpy as np
from dwt import DWTProcessor
from dct import DCTProcessor
from svd import SVDProcessor


class Embedder:
    def __init__(self, delta=16):
        self.delta = delta
        self.dwt = DWTProcessor()
        self.dct = DCTProcessor()
        self.svd = SVDProcessor()

    # ---------------------------------------------
    # Quantization
    # ---------------------------------------------
    def quantize_zero(self, value):
        return round(value / self.delta) * self.delta

    def quantize_one(self, value):
        return round(value / self.delta) * self.delta + self.delta / 2

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
    # Capacity
    # ---------------------------------------------
    def capacity(self, total_blocks):
        return total_blocks

    # ---------------------------------------------
    # Validate
    # ---------------------------------------------
    def validate_capacity(self, bits, blocks):

        if len(bits) > len(blocks):

            raise ValueError(
                f"Image capacity {len(blocks)} bits "
                f"but message requires {len(bits)} bits."
            )

    # ---------------------------------------------
    # Embed One Bit
    # ---------------------------------------------
    def embed_block(self, block, bit, energy):

        dct_block = self.dct.forward(block)

        delta = self.adaptive_delta(energy)

        positions = self.dct.mid_positions()[:3]

        for x, y in positions:

            value = dct_block[x, y]

            if bit == 0:
                value = round(value / delta) * delta
            else:
                value = round(value / delta) * delta + delta / 2

            dct_block[x, y] = value

        return self.dct.inverse(dct_block)

    # ---------------------------------------------
    # Embed Message
    # ---------------------------------------------
    def embed_message(self, selected_blocks, bits):

        self.validate_capacity(bits, selected_blocks)

        output = []

        bit_index = 0

        for item in selected_blocks:

            if bit_index >= len(bits):

                output.append(item)

                continue

            energy = item.get(
                "energy",
                self.dwt.block_energy(item["block"])
            )

            new_block = self.embed_block(
                item["block"],
                bits[bit_index],
                energy
            )

            output.append({

                "row": item["row"],

                "col": item["col"],

                "energy": energy,

                "block": new_block

            })

            bit_index += 1

        return output

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------
    def statistics(self, bits, blocks):

        return {

            "embedded_bits": len(bits),

            "available_blocks": len(blocks),

            "used_blocks": min(len(bits), len(blocks)),

            "remaining_blocks": max(0, len(blocks) - len(bits))

        }

    # ---------------------------------------------
    # Replace Blocks
    # ---------------------------------------------
    def replace_blocks(self, original_blocks, modified_blocks):

        block_map = {}

        for block in modified_blocks:

            block_map[(block["row"], block["col"])] = block

        merged = []

        for block in original_blocks:

            key = (block["row"], block["col"])

            if key in block_map:

                merged.append(block_map[key])

            else:

                merged.append(block)

        return merged

    # ---------------------------------------------
    # Merge HL
    # ---------------------------------------------
    def rebuild_hl_band(self, original_band, blocks):

        return self.dwt.merge_blocks(
            blocks,
            original_band.shape
        )

    # ---------------------------------------------
    # Update Coeff
    # ---------------------------------------------
    def update_coefficients(self, coeff, hl_band):

        coeff["HL"] = hl_band

        return coeff

    # ---------------------------------------------
    # Reconstruct
    # ---------------------------------------------
    def reconstruct_image(self, coeff):

        image = self.dwt.inverse(coeff)

        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    # ---------------------------------------------
    # Pipeline
    # ---------------------------------------------
    def embed(self, coeff, bits):

        hl = coeff["HL"]

        blocks = self.dwt.split_into_blocks(hl)

        modified = self.embed_message(blocks, bits)

        merged = self.replace_blocks(blocks, modified)

        new_hl = self.rebuild_hl_band(hl, merged)

        coeff = self.update_coefficients(coeff, new_hl)

        return self.reconstruct_image(coeff)