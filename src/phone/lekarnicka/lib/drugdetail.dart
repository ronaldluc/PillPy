import 'dart:io';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:timeago/timeago.dart' as timeago;

import 'drugitem.dart';

class DrugDetail extends StatelessWidget {
  final DrugItem item;

  const DrugDetail(this.item);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Lékárnička ‒ Detail léku')),
      body: Column(
        children: [
          Container(
            margin: EdgeInsets.only(left: 20, right: 20, top: 10),
            child: Text(
              item.name,
              style: TextStyle(fontSize: 32),
            ),
          ),
          Container(
            margin: EdgeInsets.only(top: 10),
            child: item.imagePath.contains("assets")
                ? Image.asset(item.imagePath)
                : Image.file(File(item.imagePath)),
          )
        ],
      ),
    );
  }
}
