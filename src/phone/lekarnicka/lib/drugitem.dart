import 'package:intl/intl.dart';
import 'package:timeago/timeago.dart' as timeago;


class DrugItem {
  static DateFormat dateFormat = DateFormat("dd. MM. yyyy HH:mm");
  int _id;
  String _name;
  DateTime _time;
  String _imagePath;

  DrugItem(this._name, this._time, this._imagePath);

  DrugItem.fromMap(dynamic obj) {
    // print("From map $obj");
    this._id = obj["id"];
    this._name = obj['name'];
    this._time = DateTime.parse(obj['time']);
    this._imagePath = obj['img'];
  }

  String get name => _name;
  DateTime get time => _time;
  String get imagePath => _imagePath;

  String get drugAgo {
    return timeago.format(_time, locale: "cs");
  }

  String get drugAgoLong {
    return timeago.format(_time, locale: "csl");
  }

  String get prettyTime {
    return dateFormat.format(time);
  }


  Map<String, dynamic> toMap() {
    var map = new Map<String, dynamic>();
    map["name"] = _name;
    map['time'] = _time;
    map['img'] = _imagePath;

    return map;
  }
}