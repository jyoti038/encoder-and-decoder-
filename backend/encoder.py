import numpy as np

from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from bitstream import BitStream
from crc import CRC32
from rs_codec import ReedSolomonCodec
from embed import Embedder


class Encoder:

    def __init__(self):

        self.preprocessor = ImagePreprocessor()
        self.dwt = DWTProcessor()
        self.embedder = Embedder()
        self.crc = CRC32()
        self.rs = ReedSolomonCodec()
        self.bitstream = BitStream()

    # --------------------------------------------------
    # Encode Message
    # --------------------------------------------------

    def encode(
        self,
        image_path,
        secret_message,
        output_path
    ):

        # Step 1
        data = self.preprocessor.preprocess_for_embedding(
            image_path
        )

        # Step 2
        packet = self.crc.append_crc(
            secret_message
        )

        # Step 3
        encoded = self.rs.rs.encode(packet)

        bits = self.rs.bytes_to_bits(encoded)

        print("=" * 60)
        print("Embedded Bits Length :", len(bits))
        print("Embedded First 64 Bits :", bits[:64])
        print("=" * 60)

        # Step 4
        coeff = self.dwt.forward(
            data["y_channel"]
        )

        # Step 5
        encoded_y = self.embedder.embed(
            coeff,
            bits
        )

        # ---------- DEBUG ----------
        print("Original Y Mean :", np.mean(data["y_channel"]))
        print("Embedded Y Mean :", np.mean(encoded_y))
        print("Original Y Min/Max :", np.min(data["y_channel"]), np.max(data["y_channel"]))
        print("Embedded Y Min/Max :", np.min(encoded_y), np.max(encoded_y))
        print("Difference :", np.mean(np.abs(encoded_y.astype(np.float32) - data["y_channel"].astype(np.float32))))
        print("=" * 60)
        # ---------------------------

        # Step 6
        final_image = self.preprocessor.build_final_image(
            encoded_y,
            data["cr_channel"],
            data["cb_channel"]
        )

        # Step 7
        self.preprocessor.save_image(
            final_image,
            output_path
        )

        return {
            "status": "success",
            "embedded_bits": len(bits),
            "output": output_path
        }