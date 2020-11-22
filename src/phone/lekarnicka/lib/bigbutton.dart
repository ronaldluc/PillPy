import 'dart:async';
import 'package:flutter/material.dart';

class BigButton extends StatelessWidget {
  final text;
  final callback;
  final color = Colors.green;
  final double fontSize = 30;
  final double height = 70;

  const BigButton(this.text, this.callback, {Key key})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ButtonTheme(
      height: height,
      minWidth: double.infinity,
      buttonColor: color,
      child: RaisedButton(
        child: Text(
          text,
          style: TextStyle(fontSize: fontSize, color: Colors.white),
        ),
        onPressed: callback,
      ),
    );
  }
}
