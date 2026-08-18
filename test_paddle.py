from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en')
result = ocr.ocr(r"C:\Users\Asus\Downloads\Prescription-CCC.v25i.yolov8\valid\images\tu.jpg")

for line in result[0]:
    print(line[1][0])
