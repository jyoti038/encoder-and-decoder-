from crc import CRC32

crc = CRC32()

message = "Hello Jyoti ❤️"

packet = crc.append_crc(message)

print("Packet Length")
print(len(packet))

print()

print("CRC Valid")
print(crc.verify(packet))

print()

print("Recovered")

print(crc.extract(packet))