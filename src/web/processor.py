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

from text_recognition import decode_predictions
from imutils.object_detection import non_max_suppression
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Processor(object):
    ENABLE_ZBAR_CODE = False
    ENABLE_OPENCV_QR = False
    ENABLE_OCR = True


    DRUG_LIST_FILE = "./../../data/drug_names.txt"
    EAN_TO_DRUG_LIST_FILE = "./../../data/drug_ean_to_names.txt"
    SUKL_FILE = "./../../data/benu_sukl.csv"

    OCR_DICT = "./../../data/OCR_dict.txt"
    EAST_DETECTOR = "./frozen_east_text_detection.pb"

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
        img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #img = cv2.bitwise_not(img)
        kernel = np.ones((1,1), "uint8")
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.medianBlur(img,5)

        custom_config = r'--oem 3 -l ces+en --psm 1 --user-words "' + self.OCR_DICT +  '"'
        text_ocr = pytesseract.image_to_string(img, config=custom_config)

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
                    if part_score > 0 and False:
                        print("Part match:", word, drug_processed_part, part_score, score)
 

            if score > 0 and score > best_drug_score:
                best_drug_score, best_drug = score, drug_hr

        return best_drug

    def get_ocr_text_EAST(self, image):
        (origH, origW) = image.shape[:2]
        orig = image.copy()

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (320, 320)
        rW = origW / float(newW)
        rH = origH / float(newH)


        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
        	"feature_fusion/Conv_7/Sigmoid",
        	"feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(self.EAST_DETECTOR)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        	(123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry, min_conf=0.5)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
        	# scale the bounding box coordinates based on the respective
        	# ratios
        	startX = int(startX * rW)
        	startY = int(startY * rH)
        	endX = int(endX * rW)
        	endY = int(endY * rH)

        	# in order to obtain a better OCR of the text we can potentially
        	# apply a bit of padding surrounding the bounding box -- here we
        	# are computing the deltas in both the x and y directions
        	dX = int((endX - startX) * 0.05)
        	dY = int((endY - startY) * 0.05)

        	# apply padding to each side of the bounding box, respectively
        	startX = max(0, startX - dX)
        	startY = max(0, startY - dY)
        	endX = min(origW, endX + (dX * 2))
        	endY = min(origH, endY + (dY * 2))

        	# extract the actual padded ROI
        	roi = orig[startY:endY, startX:endX]

        	# in order to apply Tesseract v4 to OCR text we must supply
        	# (1) a language, (2) an OEM flag of 4, indicating that the we
        	# wish to use the LSTM neural net model for OCR, and finally
        	# (3) an OEM value, in this case, 7 which implies that we are
        	# treating the ROI as a single line of text
        	config = (f"-l eng+ces --oem 1 --psm 7 --user-words {self.OCR_DICT}")
        	text = pytesseract.image_to_string(roi, config=config)

        	# add the bounding box coordinates and OCR'd text to the list
        	# of results
        	results.append(((startX, startY, endX, endY), text))
        
        results = sorted(results, key=lambda r:r[0][1])

        results_processed = []
        for (_, single_text) in results:
            single_text = "".join([c if ord(c) < 128 else "" for c in single_text]).strip()
            results_processed.append(single_text)

        return results_processed

    def get_and_process_OCR(self, image):
        text_ocr = self.get_ocr_text_tesseract(image)
        words_set_ocr = text_ocr.lower().split()
        
        #words_set_ocr = self.get_ocr_text_EAST(image)
        #words_set_ocr = set(filter(lambda x: len(x) > 2, words_set_ocr))

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