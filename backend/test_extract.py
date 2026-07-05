from preprocess import ImagePreprocessor
from dwt import DWTProcessor
from extract import Extractor

processor = ImagePreprocessor()

data = processor.preprocess_for_embedding("test.jpg")

y = data["y_channel"]

dwt = DWTProcessor()

coeff = dwt.forward(y)

extract = Extractor()

bits = extract.extract(
    coeff,
    total_bits=20
)

print("Recovered Bits:")
print(bits)

print()

print(extract.statistics(bits))