from os import PathLike

import sys
from typing import Tuple
import cv2
import numpy as np
from pyzbar import pyzbar
import pytesseract


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Processor(object):
    ENABLE_ZBAR_CODE = True
    ENABLE_OPENCV_QR = True
    ENABLE_OCR = True

    def __init__(self) -> None:
        self.qr_detector = cv2.QRCodeDetector() 

    def get_QR_codes_openCV(self, frame):
        detector = self.qr_detector

        _, decoded_info, _, _ = detector.detectAndDecodeMulti(frame)
        return decoded_info 

    def process_qr_code(qr_code) -> Tuple[bool, str]:
        print(f"Found QR code: {qr_code}")

        return (True, "Paralen")

    def process_EAN13_code(qr_code) -> Tuple[bool, str]:
        print(f"Found EAN13 code: {qr_code}")

        return (True, "Paralen")

    def process_EAN8_code(qr_code) -> Tuple[bool, str]:
        print(f"Found EAN8 code: {qr_code}")

        return (True, "Paralen")

    def try_detect_barcode(self, image):
        # Maybe: https://www.mdpi.com/2076-3417/9/16/3268/htm
        # Adapted from: https://www.pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # equalize lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # edge enhancement
        edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                                 ksize = 3, scale = 1, delta = 0)
        # retval = cv2.imwrite("edge_enh.jpg", edge_enh)
        
        # bilateral blur, which keeps edges
        blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)
        
        # use simple thresholding. adaptive thresholding might be more robust
        (_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
        # retval = cv2.imwrite("thresh.jpg", thresh)
        
        # do some morphology to isolate just the barcode blob
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)
        # retval = cv2.imwrite("closed.jpg", closed)
        
        # find contours left in the image
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # retval = cv2.imwrite("found.jpg", image)

        # actually gets you rotated rect instead of quadrilateral
        return box

    def get_and_process_OCR(self, image):
        custom_config = r'--oem 3 --psm 1 --user-words "./data/drug_names.txt"'
        text_ocr = pytesseract.image_to_string(image, config=custom_config)
        print(text_ocr)
        
        return (True, "Paralen")

    def process(self, img_path: PathLike) -> Tuple[bool, str]:

        frame = cv2.imread(img_path)

        width, height = int(frame.shape[1]), int(frame.shape[0])
        print(f"Uploaded w:{width}, h:{height}")

        # self.try_detect_barcodes(frame)
        # when we get barcode bounding box -> transform the image to get nice orthogonal view (i.e apply transf. so that b.b is non-rotated rect)
        
        if self.ENABLE_ZBAR_CODE:
            codes_zbar = pyzbar.decode(frame)
            for code in codes_zbar:
                data, type = code.data.decode("utf-8"), code.type

                if type == "QRCODE":
                    succ, name = self.process_qr_code(data)
                elif type == "EAN13":
                    succ, name = self.process_EAN13_code(data)
                elif type == "EAN8":
                    succ, name = self.process_EAN13_code(data)
                else:
                    succ, name = False, None

                if succ:
                    return (succ, name)

        if self.ENABLE_OPENCV_QR:
            qr_codes = self.get_QR_codes_openCV(frame)
            for data in qr_codes:
                succ, name = self.process_qr_code(data)
                if succ:
                    return (succ, name)
        
        if self.ENABLE_OCR:
            succ, name = self.get_and_process_OCR(frame)
            if succ:
                return (succ, name)

        ##         dim = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))

        #frame = frame[(frame.shape[0] - height) // 2: (frame.shape[0] + height) // 2,
                #(frame.shape[1] - width) // 2:  (frame.shape[1] + width) // 2]
        ##         frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        ##         crop_img = img[y:y+h, x:x+w]
        #cv2.imwrite('cropped.jpg', frame)

        #img = open_image('cropped.jpg')
        #cat, _, prob = self.model.predict(img)

        return (False, None)


if __name__ == "__main__":
    processor = Processor()
    response = processor.process(sys.argv[1])
    print(response)