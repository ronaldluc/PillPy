import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:timeago/timeago.dart' as timeago;



import 'scanner.dart';

void main() async {
  timeago.setLocaleMessages("cs", timeago.CsShortMessages());
  timeago.setLocaleMessages("csl", timeago.CsMessages());

  WidgetsFlutterBinding.ensureInitialized();

  // Obtain a list of the available cameras on the device.
  final cameras = await availableCameras();

  // Get a specific camera from the list of available cameras.
  final firstCamera = cameras.first;

  runApp(
    MaterialApp(
      title: 'Lékárnička',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: DrugScanner(
        camera: firstCamera,
      ),
      navigatorObservers: [routeObserver],
    ),
  );
}
