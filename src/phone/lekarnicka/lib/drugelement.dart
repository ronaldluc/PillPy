import 'dart:async';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:timeago/timeago.dart' as timeago;

import 'drugitem.dart';
import 'drugdetail.dart';

class DrugElement extends StatelessWidget {
  final DrugItem item;



  const DrugElement(this.item);

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Row(
        // mainAxisAlignment: MainAxisAlignment.end,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            // alignment: Alignment.centerLeft,
            child: Text(
              item.name,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 24),
            ),
          ),
          Container(
            margin: EdgeInsets.only(left: 10),
            // alignment: Alignment.centerRight,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  item.drugAgo,
                  textAlign: TextAlign.right,
                  style: TextStyle(fontSize: 20),
                ),
                Text(
                  item.prettyTime,
                  textAlign: TextAlign.right,
                ),
              ],
            ),
          ),
        ],
      ),
      onTap: () => _showDrugDetail(context),
    );
  }

  void _showDrugDetail(context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => DrugDetail(item)),
    );
  }
}
