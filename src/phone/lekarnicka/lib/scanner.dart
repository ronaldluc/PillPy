import 'dart:async';
import 'dart:io';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:lekarnicka/drugitem.dart';
import 'package:lekarnicka/druglist.dart';
import 'package:path/path.dart' show join;
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:wakelock/wakelock.dart';
import 'package:soundpool/soundpool.dart';
import 'package:flutter/services.dart';
import 'package:semaphore/semaphore.dart';

import 'bigbutton.dart';
import 'querry.dart';
import 'popup.dart';

final RouteObserver<PageRoute> routeObserver = RouteObserver<PageRoute>();

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

class DrugScannerState extends State<DrugScanner>
    with WidgetsBindingObserver, RouteAware {
  AppLifecycleState _notification;

  QueryCtr _query;
  Soundpool soundPool;
  Map<String, int> sounds = {};

  CameraController _controller;
  Future<void> _initializeControllerFuture;

  static bool startCalls = true;
  bool sendCalls = true;
  bool showingPopup = false;
  bool get stopSending => (!sendCalls || showingPopup);
  final cameraLock = LocalSemaphore(1);

  static double buttonHeight = 80;
  static double fontSize = 30;
  Timer photoTimer;

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    print("New state: $state");
    _notification = state;
    if (state == AppLifecycleState.paused) {
      sendCalls = false;
      Wakelock.disable();
    } else if (state == AppLifecycleState.resumed) {
      sendCalls = true;
      Wakelock.enable();
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    routeObserver.subscribe(this, ModalRoute.of(context));
  }

  @override
  void didPopNext() {
    // Covering route was popped off the navigator.
    sendCalls = true;
    print("!!!Pop!!!");
    Wakelock.enable();
  }

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addObserver(this);
    _query = QueryCtr();
    soundPool = Soundpool(streamType: StreamType.notification);
    print(soundPool);
    _loadSounds();

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
        Duration(seconds: 8),
        _runRecognition,
      );
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    routeObserver.unsubscribe(this);
    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  _loadSounds() async {
    final ByteData soundData = await rootBundle.load("sounds/lek_pridan.m4a");
    print("aaa $soundPool $soundData");
    sounds["lek_pridan"] = await soundPool.load(soundData);
  }

  void _runRecognition(Timer t) async {
    if (stopSending) {
      return;
    }

    print("------- Running recognition");
    var photo = await _takePhoto();
    if (photo == null) {
      return;
    }

    // try {
    //   var data = await QrCodeToolsPlugin.decodeFrom(photo);
    //   print("QR: $data");
    // } catch (PlatformException) {
    //   print("Exception");
    // }

    var status = await _sendImage(photo);
    if (status["success"]) {
      print("Setting status $status");
      // soundPool.play(sounds["lek_pridan"]);
      _addDrug(status["name"], photo);
    } else if (status == "nothing") {
      File(photo).delete();
      print("File $photo deleted.");
      return;
    }
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
              () => _pushScreen(context, DrugList(_query)),
            ),
          ),
          Container(
            alignment: Alignment.bottomCenter,
            padding: EdgeInsets.all(10),
            child: BigButton(
              "Přidat lék manuálně",
              _addManually,
            ),
          ),
        ],
      ),
    );
  }

  _addDrug(name, image) async {
    print("adding drug $name");
    var item = DrugItem(name, DateTime.now(), image);
    var res = await _query.addDrug(item);
    _drugPopup(item);
    soundPool.play(sounds["lek_pridan"]);
  }

  _addManually() async {
    var image = await _takePhoto();
    _addDrug("Neznámé léčivo", image);
  }

  _drugPopup(DrugItem item) {
    if (showingPopup) {
      _closePopup();
    }
    showingPopup = true;

    showDialog(
        context: context,
        builder: (BuildContext dialogContext) {
          return WillPopScope(
            child: PopupDrug(item.name, item.prettyTime, item.imagePath),
            onWillPop: () async {
              showingPopup = false;
              return true;
            },
          );
        }).then((value) => print("Closing"));

    Timer(Duration(seconds: 2), _closePopup);
  }

  _closePopup() {
    if (showingPopup) {
      Navigator.pop(context);
    }
    showingPopup = false;
  }

  _pushScreen(context, screen) {
    sendCalls = false;
    Wakelock.disable();
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => screen),
    );
  }

  Future<String> _takePhoto() async {
    try {
      await cameraLock.acquire();
      await _initializeControllerFuture;

      final path = join(
        (await getTemporaryDirectory()).path,
        '${DateTime.now()}.png',
      );

      await _controller.takePicture(path);
      cameraLock.release();
      return path;
    } catch (e) {
      print(e);
      cameraLock.release();
      return null;
    }
  }

  Future<Map<String, dynamic>> _sendImage(path) async {
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
      var response = await req.timeout(const Duration(seconds: 10));
      print("Got response $response");
      if (response.statusCode == 200) {
        return jsonDecode(await response.stream.bytesToString());
      }
      return {"success": false, "error": "Bad status: ${response.statusCode}"};
    } on TimeoutException catch (_) {
      print("Request timeout");
      return {"success": false, "error": "Timeout"};
    } on SocketException catch (_) {
      print("Socket error");
      return {"success": false, "error": "Socket error"};
    }
  }
}
