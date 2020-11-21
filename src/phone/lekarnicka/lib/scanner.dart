import 'dart:async';
import 'dart:io';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:lekarnicka/druglist.dart';
import 'package:path/path.dart' show join;
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import 'bigbutton.dart';

// A screen that allows users to take a picture using a given camera.
class DrugScanner extends StatefulWidget {
  final CameraDescription camera;

  const DrugScanner({
    Key key,
    @required this.camera,
  }) : super(key: key);

  @override
  DrugScannerState createState() => DrugScannerState();
}

class DrugScannerState extends State<DrugScanner> {
  CameraController _controller;
  Future<void> _initializeControllerFuture;

  static bool startCalls = true;

  static double buttonHeight = 80;
  static double fontSize = 30;
  bool isAccepting = true;
  Timer photoTimer;

  @override
  void initState() {
    super.initState();
    // To display the current output from the Camera,
    // create a CameraController.
    _controller = CameraController(
      // Get a specific camera from the list of available cameras.
      widget.camera,
      // Define the resolution to use.
      ResolutionPreset.high,
      enableAudio: false,
    );

    print("Initing camera");
    // Next, initialize the controller. This returns a Future.
    _initializeControllerFuture = _controller.initialize();

    if (startCalls) {
      Timer(Duration(seconds: 1), () => _runRecognition(null));
      // starting timer
      photoTimer = Timer.periodic(
        Duration(seconds: 10),
        _runRecognition,
      );
    }
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  void _runRecognition(Timer t) async {
    print("------- Running recognition");
    var photo = await _takePhoto();
    if (photo == null) {
      return;
    }

    var status = await _sendImage(photo);
    if (status == null) {
      return;
    } else if (status == "nothing") {
      File(photo).delete();
      print("File $photo deleted.");
      return;
    }

    print("Setting status $status");
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Lékárnička')),
      body: Stack(
        children: [
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                // If the Future is complete, display the preview.
                return CameraPreview(_controller);
              } else {
                // Otherwise, display a loading indicator.
                return Center(child: CircularProgressIndicator());
              }
            },
          ),
          Container(
            alignment: Alignment.topCenter,
            padding: EdgeInsets.all(10),
            child: BigButton(
              "seznam léků",
              () => {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => DrugList()),
                )
              },
            ),
          ),
          Container(
            alignment: Alignment.bottomCenter,
            padding: EdgeInsets.all(10),
            child: BigButton(
              "Přidat lék manuálně",
              () => null,
            ),
          ),
        ],
      ),
    );
  }

  Future<String> _takePhoto() async {
    try {
      await _initializeControllerFuture;

      final path = join(
        (await getTemporaryDirectory()).path,
        '${DateTime.now()}.png',
      );

      await _controller.takePicture(path);
      return path;
    } catch (e) {
      print(e);
      return null;
    }
  }

  Future<String> _sendImage(path) async {
    print("Sending image $path");

    var url = Uri.parse("http://baiku.cz:4444");
    var request = new http.MultipartRequest("POST", url);
    request.fields['user'] = 'someone@somewhere.com';
    request.files.add(await http.MultipartFile.fromPath(
      'upload',
      path,
      contentType: MediaType('image', 'png'),
    ));
    var req = request.send();
    print("Request sent");
    try {
      var response = await req.timeout(const Duration(seconds: 5));
      print("Got response $response");
      if (response.statusCode == 200) {
        return response.stream.bytesToString();
      }
      return "nothing";
    } on TimeoutException catch (_) {
      print("Request timeout");
      return "nothing";
    } on SocketException catch (_) {
      print("Socket error");
      return "nothing";
    }
  }
}
