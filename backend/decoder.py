from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from extract import Extractor
from rs_codec import ReedSolomonCodec
from crc import CRC32


class Decoder:

    def __init__(self):

        self.preprocessor = ImagePreprocessor()
        self.dwt = DWTProcessor()
        self.extractor = Extractor()
        self.rs = ReedSolomonCodec()
        self.crc = CRC32()

    # --------------------------------------------------
    # Decode Secret Message
    # --------------------------------------------------
    def decode(self, image_path, total_bits):

        # Step 1
        data = self.preprocessor.preprocess_for_embedding(
            image_path
        )

        # Step 2
        coeff = self.dwt.forward(
            data["y_channel"]
        )

        # Step 3
        bits = self.extractor.extract(
            coeff,
            total_bits
        )
        print("Recovered Full Bits:", bits)
        print("Recovered First 64:", bits[:64])

        if len(bits) % 8 != 0:
            bits = bits[: len(bits) - (len(bits) % 8)]

        if len(bits) == 0:
            raise ValueError("Recovered bitstream is empty.")

        print("Recovered Bits :", len(bits))
        print("First 64 bits :", "".join(map(str, bits[:64])))

        encoded_bytes = self.rs.bits_to_bytes(bits)

        decoded_packet = None

        for end in range(len(encoded_bytes), 3, -1):
            candidate = encoded_bytes[:end]

            try:
                packet = self.rs.rs.decode(candidate)

                if isinstance(packet, tuple):
                    packet = packet[0]

                if self.crc.verify(packet):
                    decoded_packet = packet
                    break

            except Exception:
                continue

        if decoded_packet is None:
            raise ValueError("CRC verification failed.")

        secret = self.crc.extract(decoded_packet)

        return {
            "status": "success",
            "secret_message": secret
        }


# --------------------------------------------------
# Wrapper Function
# --------------------------------------------------
def decode_image(image_path, total_bits):
    decoder = Decoder()
    return decoder.decode(image_path, total_bits)