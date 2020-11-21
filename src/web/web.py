#!/usr/bin/env python
model_name = 'silicon-1.1'

"""Extend Python's built in HTTP server to save files
curl or wget can be used to send files with options similar to the following
  curl -X PUT --upload-file somefile.txt http://localhost:8000
  wget -O- --method=PUT --body-file=somefile.txt http://localhost:8000/somefile.txt
__Note__: curl automatically appends the filename onto the end of the URL so
the path can be omitted.
"""

import os
from pathlib import Path

from processor import Processor
import http.server as server

import cgi
import json
import uuid



class HTTPRequestHandler(server.BaseHTTPRequestHandler):
    """Extend SimpleHTTPRequestHandler to handle PUT requests"""

    def __init__(self, request, client_address, server):
        self.tmp_folder_path = self.init_tmp_folder()
        self.processor = Processor()

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

        success, name = self.processor.process(file_path)
        response = json.dumps({"success": success, "name": name})

        self.send_response(200, 'Created')
        self.end_headers()

        print(f"Response: {response}")

        self.wfile.write(response.encode('utf-8'))
        
        # cleanup
        #os.remove(file_path)


if __name__ == '__main__':
    server.test(HandlerClass=HTTPRequestHandler, port=4444)
