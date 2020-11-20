#!/usr/bin/env python
model_name = 'silicon-1.1'

"""Extend Python's built in HTTP server to save files
curl or wget can be used to send files with options similar to the following
  curl -X PUT --upload-file somefile.txt http://localhost:8000
  wget -O- --method=PUT --body-file=somefile.txt http://localhost:8000/somefile.txt
__Note__: curl automatically appends the filename onto the end of the URL so
the path can be omitted.
"""
from time import sleep

import cv2
import numpy as np
import os

import multiprocessing
from pathlib import Path

from glob import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import http.server as server

import random
import string


class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
    """Extend SimpleHTTPRequestHandler to handle PUT requests"""

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_POST(self):
        """Save a file following a HTTP POST request"""
        #filename = "pes"

        #file_length = int(self.headers['Content-Length'])
        #with open(filename, 'wb') as output_file:
            #output_file.write(self.rfile.read(file_length))

        #frame = cv2.imread(filename)

        #width = int(frame.shape[1] * 0.5)
        #height = int(frame.shape[0] * 0.5)
        ##         dim = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))

        #frame = frame[(frame.shape[0] - height) // 2: (frame.shape[0] + height) // 2,
                #(frame.shape[1] - width) // 2:  (frame.shape[1] + width) // 2]
        ##         frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        ##         crop_img = img[y:y+h, x:x+w]
        #cv2.imwrite('cropped.jpg', frame)

        #img = open_image('cropped.jpg')
        #cat, _, prob = self.model.predict(img)
        self.send_response(200, 'Created')
        self.end_headers()
        #         reply_body = f'{cat.strip(['% \n'])}\n'
        reply_body = "From Server!"
        print(reply_body)
        #print(cat, prob)
        self.wfile.write(reply_body.encode('utf-8'))


if __name__ == '__main__':
    server.test(HandlerClass=HTTPRequestHandler, port=4444)
