from preprocess import ImagePreprocessor

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

rgb = processor.build_final_image(

    data["y_channel"],

    data["cb_channel"],

    data["cr_channel"]

)

processor.save_image(

    "reconstructed.png",

    rgb

)

print("Done")