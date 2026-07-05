from rs_codec import ReedSolomonCodec

codec = ReedSolomonCodec()

message = "Hello Jyoti ❤️"

print("Original")
print(message)

encoded = codec.encode(message)

print()
print("Encoded Bytes")
print(encoded)

bits = codec.encode_bits(message)

print()
print("Total Bits")
print(len(bits))

decoded = codec.decode(encoded)

print()
print("Decoded")
print(decoded)