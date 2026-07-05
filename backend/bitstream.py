"""
bitstream.py

Converts secret messages into binary bit streams and back.

Pipeline

Text
↓

UTF-8 Bytes
↓

32-bit Length Header
↓

Binary Bit Stream

Decoder

Bit Stream
↓

Length Header
↓

Bytes
↓

UTF-8 Text
"""

from typing import List


class BitStream:

    HEADER_SIZE = 32

    # ---------------------------------
    # TEXT -> BYTES
    # ---------------------------------

    @staticmethod
    def text_to_bytes(text: str) -> bytes:
        return text.encode("utf-8")

    # ---------------------------------
    # BYTES -> TEXT
    # ---------------------------------

    @staticmethod
    def bytes_to_text(data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")

    # ---------------------------------
    # INTEGER -> FIXED LENGTH BITS
    # ---------------------------------

    @staticmethod
    def int_to_bits(number: int, bit_length: int) -> List[int]:

        return [
            int(bit)
            for bit in format(number, f"0{bit_length}b")
        ]

    # ---------------------------------
    # FIXED LENGTH BITS -> INTEGER
    # ---------------------------------

    @staticmethod
    def bits_to_int(bits: List[int]) -> int:

        binary = "".join(str(b) for b in bits)

        return int(binary, 2)

    # ---------------------------------
    # BYTES -> BITS
    # ---------------------------------

    @staticmethod
    def bytes_to_bits(data: bytes) -> List[int]:

        bits = []

        for byte in data:

            bits.extend(
                BitStream.int_to_bits(byte, 8)
            )

        return bits

    # ---------------------------------
    # BITS -> BYTES
    # ---------------------------------

    @staticmethod
    def bits_to_bytes(bits: List[int]) -> bytes:

        if len(bits) % 8 != 0:
            raise ValueError(
                "Bit stream length must be divisible by 8."
            )

        output = bytearray()

        for i in range(0, len(bits), 8):

            value = BitStream.bits_to_int(
                bits[i:i + 8]
            )

            output.append(value)

        return bytes(output)

    # ---------------------------------
    # CREATE COMPLETE BIT STREAM
    # ---------------------------------

    @staticmethod
    def create_bitstream(message: str) -> List[int]:

        message_bytes = BitStream.text_to_bytes(message)

        message_length = len(message_bytes)

        header = BitStream.int_to_bits(
            message_length,
            BitStream.HEADER_SIZE
        )

        payload = BitStream.bytes_to_bits(
            message_bytes
        )

        return header + payload

    # ---------------------------------
    # READ HEADER
    # ---------------------------------

    @staticmethod
    def extract_length(bits: List[int]) -> int:

        if len(bits) < BitStream.HEADER_SIZE:
            raise ValueError("Header missing.")

        return BitStream.bits_to_int(
            bits[:BitStream.HEADER_SIZE]
        )

    # ---------------------------------
    # EXTRACT MESSAGE
    # ---------------------------------

    @staticmethod
    def extract_message(bits: List[int]) -> str:

        length = BitStream.extract_length(bits)

        start = BitStream.HEADER_SIZE

        end = start + (length * 8)

        payload = bits[start:end]

        message_bytes = BitStream.bits_to_bytes(
            payload
        )

        return BitStream.bytes_to_text(
            message_bytes
        )

    # ---------------------------------
    # VALIDATE MESSAGE SIZE
    # ---------------------------------

    @staticmethod
    def required_bits(message: str) -> int:

        return len(
            BitStream.create_bitstream(message)
        )

    # ---------------------------------
    # DEBUG
    # ---------------------------------

    @staticmethod
    def print_statistics(message: str):

        bits = BitStream.create_bitstream(message)

        print("=" * 40)
        print("Message:", message)
        print("Characters :", len(message))
        print("Bits :", len(bits))
        print("Header :", BitStream.HEADER_SIZE)
        print("Payload :", len(bits) - BitStream.HEADER_SIZE)
        print("=" * 40)