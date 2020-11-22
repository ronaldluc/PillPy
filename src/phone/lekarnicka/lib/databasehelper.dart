import 'dart:io';
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper.internal();
  factory DatabaseHelper() => _instance;
  static Database _db;
  Future<Database> get db async {
    // print("In DB future");
    if (_db != null) {
      return _db;
    }
    _db = await initDb();
    return _db;
  }

  DatabaseHelper.internal();
  initDb() async {
    print("Initing DB");
    Directory documentDirectory = await getApplicationDocumentsDirectory();
    String path = join(documentDirectory.path, "data_flutter.db");

    print("Got Directory DB");
    // Only copy if the database doesn't exist
    if (true || FileSystemEntity.typeSync(path) == FileSystemEntityType.notFound) {
      // Load database from asset and copy
      // print("AAA");

      ByteData data = await rootBundle.load(join('assets', 'data_flutter.db'));
      List<int> bytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      // Save copied asset to documents
      // print("Saving DB");
      await File(path).writeAsBytes(bytes);
    }
    print("Opening DB");
    var ourDb = await openDatabase(path);
    // print("DB opened");
    return ourDb;
  }
}
