import 'dart:io';
import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';
import 'package:intl/intl.dart';
import 'package:timeago/timeago.dart' as timeago;

import 'drugitem.dart';

class DrugDetail extends StatelessWidget {
  final DrugItem item;

  const DrugDetail(this.item);

  get image {
    return item.imagePath.contains("assets")
        ? AssetImage(item.imagePath)
        : FileImage(File(item.imagePath));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Lékárnička ‒ Detail léku')),
      body: Column(
        children: [
          Container(
            margin: EdgeInsets.only(left: 20, right: 20, top: 20, bottom: 10),
            child: Text(
              item.name,
              style: TextStyle(fontSize: 32),
            ),
          ),
          Divider(
            color: Colors.black,
          ),
          Container(
            margin: EdgeInsets.only(top: 10),
            child: Text(
              item.drugAgoLong,
              textAlign: TextAlign.right,
              style: TextStyle(fontSize: 20),
            ),
          ),
          Text(
            item.prettyTime,
            textAlign: TextAlign.right,
          ),
          Expanded(
            child: Container(
              margin: EdgeInsets.only(top: 10),
              child: PhotoView(
                backgroundDecoration: BoxDecoration(color: Colors.white),
                imageProvider: image,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
