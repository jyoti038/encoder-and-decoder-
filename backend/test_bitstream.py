from bitstream import BitStream

message = "Hello Jyoti ❤️"

bits = BitStream.create_bitstream(message)

print("Total Bits:", len(bits))

decoded = BitStream.extract_message(bits)

print(decoded)

BitStream.print_statistics(message)