from os import PathLike
from fuzzywuzzy import fuzz

import sys
from typing import Tuple
import cv2
import numpy as np
from pyzbar import pyzbar
import pytesseract
import re
import csv


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Processor(object):
    ENABLE_ZBAR_CODE = True
    ENABLE_OPENCV_QR = True
    ENABLE_OCR = True


    DRUG_LIST_FILE = "./../../data/drug_names.txt"
    EAN_TO_DRUG_LIST_FILE = "./../../data/drug_ean_to_names.txt"
    SUKL_FILE = "./../../data/benu_sukl.csv"

    OCR_DICT = "./../../data/OCR_dict.txt"

    @staticmethod
    def __until_first_lower_case(string):
        for i, c in enumerate(string):
            if c != c.upper():
                return string[:i]
        return string

    def __init__(self) -> None:
        self.qr_detector = cv2.QRCodeDetector()

        list_of_words_for_ocr = []

        drug_list = []
        drug_list_processed = []
        with open(self.DRUG_LIST_FILE) as drug_list_f:
            drug_list = drug_list_f.readlines()
            for drug in drug_list:
                drug_list_processed.append(list(filter(lambda x: len(x) > 2, drug.lower().split())))

                list_of_words_for_ocr += drug.lower().split()
                list_of_words_for_ocr.append(drug)

        ean_to_drugs_dict = {}
        with open(self.EAN_TO_DRUG_LIST_FILE) as drug_list_f:
            ean_to_drugs_list = drug_list_f.readlines()
            for ean_to_drug in ean_to_drugs_list:
                ean, drug = ean_to_drug.split(" ", 1)
                ean_to_drugs_dict[ean] = drug

        sukl_to_drugs_dict = {}

        with open(self.SUKL_FILE) as sukl_f:
            reader = csv.DictReader(sukl_f)
            for row in reader:
                sukl, name, ean = row["sukl_code"], row["name"], row["ean_code"]
                pure_name = self.__until_first_lower_case(name)

                drug_list.append(name)
                drug_list_processed.append(list(filter(lambda x: len(x) > 2, pure_name.lower().split())))

                ean_to_drugs_dict[ean] = name
                sukl_to_drugs_dict[sukl] = name
                list_of_words_for_ocr += [pure_name, name] + name.split()

        self.drug_list = drug_list                          # List of drugs full names
        self.drug_list_processed = drug_list_processed      # List of drug names splitted in lowercase without generic fluff (potahované tablety, ...), only words longer than 2

        self.ean_to_drugs_dict = ean_to_drugs_dict          # EAN to full name
        self.sukl_to_drugs_dict = sukl_to_drugs_dict        # SUKL to full name

        with open(self.OCR_DICT, "w") as f_ocr_dict:
            f_ocr_dict.writelines(map(lambda x: x+"\n", list_of_words_for_ocr))

    def get_QR_codes_openCV(self, frame):
        detector = self.qr_detector

        _, decoded_info, _, _ = detector.detectAndDecodeMulti(frame)
        return decoded_info 

    def process_qr_code(self, code) -> Tuple[bool, str]:
        print(f"Found QR code: {code}")
        processed_qr_string = set(code.split())
        best_drug = self.find_intersection_in_drugs_names_list(processed_qr_string)

        if best_drug:
            print(f"QR match: {best_drug}")
            return (True, best_drug)
        else:
            print(f"QR no-match")
            return (False, None)

    def process_EAN_code(self, code) -> Tuple[bool, str]:
        print(f"Found EAN code: {code}")

        if code in self.ean_to_drugs_dict:
            best_drug = self.ean_to_drugs_dict[code]
            print(f"EAN match: {best_drug}")
            return (True, best_drug)

        print(f"EAN no-match")
        return (False, None)

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

    def get_ocr_text_tesseract(self, image):
        custom_config = r'--oem 3 -l ces+en --psm 1 --user-words "' + self.OCR_DICT +  '"'
        text_ocr = pytesseract.image_to_string(image, config=custom_config)

        return text_ocr

    def find_intersection_in_drugs_names_list(self, set_of_words):
        best_drug, best_drug_score = None, -1

        for i, drug_hr in enumerate(self.drug_list):
            score = -1
            drug_processed = self.drug_list_processed[i]

            for word in set_of_words:
                for wi, drug_processed_part in enumerate(drug_processed):
                    part_score = fuzz.ratio(word, drug_processed_part)
                    if part_score < 50: part_score = 0              # Disregard non-matches

                    if wi == 0:                                     # First few words are more important
                        part_score *= 10
                        if part_score > 850: part_score *= 1000     # High first word match -> most important

                    elif wi == 1: part_score *= 3
                    elif wi == 2: part_score *= 1.2

                    score += part_score
                    if part_score > 0:
                        print("AAAAA", word, drug_processed_part, part_score, score)
 

            if score > 0 and score > best_drug_score:
                best_drug_score, best_drug = score, drug_hr

        return best_drug

    def get_and_process_OCR(self, image):
        text_ocr = self.get_ocr_text_tesseract(image)
        words_set_ocr = text_ocr.lower().split()
        words_set_ocr = set(filter(lambda x: len(x) > 2, words_set_ocr))

        print(f"Found OCR {words_set_ocr}")
        best_drug = self.find_intersection_in_drugs_names_list(words_set_ocr)

        if best_drug:
            print(f"OCR match: {best_drug}")
            return (True, best_drug)
        else:
            print(f"OCR no-match")
            return (False, None)

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
                    succ, name = self.process_EAN_code(data)
                elif type == "EAN8":
                    succ, name = self.process_EAN_code(data)
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