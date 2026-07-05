from reedsolo import RSCodec, ReedSolomonError


class ReedSolomonCodec:

    def __init__(self, ecc_bytes=8):
        """
        ecc_bytes = Number of error correction bytes.
        32 is a good starting point for print-scan robustness.
        """
        self.ecc_bytes = ecc_bytes
        self.rs = RSCodec(ecc_bytes)

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------
    def encode(self, message: str) -> bytes:

        if not isinstance(message, str):
            raise TypeError("Message must be string.")

        data = message.encode("utf-8")

        return self.rs.encode(data)

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------
    def decode(self, encoded_data: bytes) -> str:

        try:

            decoded = self.rs.decode(encoded_data)

            # reedsolo may return tuple in newer versions
            if isinstance(decoded, tuple):
                decoded = decoded[0]

            return decoded.decode("utf-8")

        except ReedSolomonError:
            raise ValueError("Unable to recover message.")

    # --------------------------------------------------
    # Bytes → Bits
    # --------------------------------------------------
    @staticmethod
    def bytes_to_bits(data):

        bits = []

        for byte in data:

            bits.extend(
                [int(bit) for bit in format(byte, "08b")]
            )

        return bits

    # --------------------------------------------------
    # Bits → Bytes
    # --------------------------------------------------
    @staticmethod
    def bits_to_bytes(bits):

        if len(bits) % 8 != 0:
            raise ValueError(
                "Bit length must be multiple of 8."
            )

        output = bytearray()

        for i in range(0, len(bits), 8):

            byte = bits[i:i+8]

            output.append(
                int(
                    "".join(map(str, byte)),
                    2
                )
            )

        return bytes(output)

    # --------------------------------------------------
    # Encode directly to bitstream
    # --------------------------------------------------
    def encode_bits(self, message):

        encoded = self.encode(message)

        return self.bytes_to_bits(encoded)

    # --------------------------------------------------
    # Decode from bitstream
    # --------------------------------------------------
    def decode_bits(self, bits):

        data = self.bits_to_bytes(bits)

        return self.decode(data)