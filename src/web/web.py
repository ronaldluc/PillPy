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

import cgi
import json
import uuid

class HTTPRequestHandler(server.BaseHTTPRequestHandler):
    """Extend SimpleHTTPRequestHandler to handle PUT requests"""

    def __init__(self, request, client_address, server):
        self.tmp_folder_path = self.init_tmp_folder()
        super().__init__(request, client_address, server)

    def init_tmp_folder(self):
        file_folder = "tmp"

        folder_path = Path(os.path.join(os.getcwd(), file_folder))
        folder_path.mkdir(exist_ok=True)

        return folder_path

    def get_data_from_multipart(self):
        # Adapted from: https://gist.github.com/MFry/90382082f9a65eceabd007ee7182af92
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")

        assert ctype == 'multipart/form-data'

        fields = cgi.parse_multipart(self.rfile, pdict)
        upload_field = fields.get('upload')
        assert len(upload_field) == 1

        return upload_field[0]

    def do_POST(self):
        """Save a file following a HTTP POST request"""
        file_name = f"{uuid.uuid4()}.png"
        file_path = os.path.join(self.tmp_folder_path, file_name)

        msg_raw = self.get_data_from_multipart()
        with open(file_path, 'wb') as output_file:
            output_file.write(msg_raw)

        frame = cv2.imread(file_path)

        width = int(frame.shape[1])
        height = int(frame.shape[0])

        print(f"Uploaded w:{width}, h:{height}")
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

        # reply_body = json.dumps({"success": False, "name": None})   # {"success": false, "name": null}
        reply_body = json.dumps({"success": True, "name": "Paralen"}) # {"success": true, "name": "Paralen"}
        print(f"Response: {reply_body}")

        self.wfile.write(reply_body.encode('utf-8'))
        
        # cleanup
        os.remove(file_path)


if __name__ == '__main__':
    server.test(HandlerClass=HTTPRequestHandler, port=4444)
