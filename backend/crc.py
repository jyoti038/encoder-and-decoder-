import zlib


class CRC32:

    def __init__(self):
        pass

    # ----------------------------------------
    # Generate CRC
    # ----------------------------------------

    def generate(self, data: bytes) -> int:

        return zlib.crc32(data) & 0xffffffff

    # ----------------------------------------
    # Append CRC
    # ----------------------------------------

    def append_crc(self, message: str) -> bytes:

        data = message.encode("utf-8")

        crc = self.generate(data)

        crc_bytes = crc.to_bytes(
            4,
            byteorder="big"
        )

        return data + crc_bytes

    # ----------------------------------------
    # Verify CRC
    # ----------------------------------------

    def verify(self, packet: bytes):

        if len(packet) < 4:
            return False

        data = packet[:-4]

        received = int.from_bytes(
            packet[-4:],
            byteorder="big"
        )

        calculated = self.generate(data)

        return received == calculated

    # ----------------------------------------
    # Extract Message
    # ----------------------------------------

    def extract(self, packet: bytes):

        if not self.verify(packet):
            raise ValueError(
                "CRC Verification Failed."
            )

        return packet[:-4].decode(
            "utf-8",
            errors="ignore"
        )