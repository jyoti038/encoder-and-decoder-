from encoder import Encoder

encoder = Encoder()

result = encoder.encode(

    image_path="test.jpg",

    secret_message="Hello Jyoti ❤️",

    output_path="encoded.png"

)

print(result)