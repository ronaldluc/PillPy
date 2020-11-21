from os import PathLike

import sys
from typing import Tuple
import cv2
from pyzbar import pyzbar

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Processor(object):
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
        print(f"Found QR code: {qr_code}")

        return (True, "Paralen")


    def process(self, img_path: PathLike) -> Tuple[bool, str]:

        frame = cv2.imread(img_path)

        width = int(frame.shape[1])
        height = int(frame.shape[0])

        codes_zbar = pyzbar.decode(frame)
        for code in codes_zbar:
            data, type = code.data.decode("utf-8"), code.type

            if type == "QRCODE":
                succ, name = self.process_qr_code(data)
            elif type == "EAN13":
                succ, name = self.process_EAN13_code(data)
            else:
                succ, name = False, None

            if succ:
                return (succ, name)

        qr_codes = self.get_QR_codes_openCV(frame)
        for data in qr_codes:
            succ, name = self.process_qr_code(data)
            if succ:
                return (succ, name)
        
        print(f"Uploaded w:{width}, h:{height}")
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