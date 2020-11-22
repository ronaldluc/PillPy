import 'dart:ui';
import 'dart:io';

import 'package:flutter/material.dart';

class PopupDrug extends StatelessWidget {
  final String title;
  final String text;
  final String image;

  PopupDrug(this.title, this.text, this.image);

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Container(
        child: Text(
          this.title,
          style: Theme.of(context).textTheme.headline5,
          textAlign: TextAlign.center,
        ),
      ),
      content: Container(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image(
              image: FileImage(File(image)),
              fit: BoxFit.fitWidth,
              height: 300,
            ),
            Container(
              child: Text(
                this.text,
                style: Theme.of(context).textTheme.bodyText1,
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
      actions: [],
    );
  }
}
