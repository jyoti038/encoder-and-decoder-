from decoder import Decoder

decoder = Decoder()

result = decoder.decode(

    image_path="encoded.png",

    total_bits=384

)

print(result)