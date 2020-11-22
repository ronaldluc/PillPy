import 'dart:async';

import 'databasehelper.dart';
import 'drugitem.dart';

class QueryCtr {
  DatabaseHelper con = DatabaseHelper();
  Future<List<DrugItem>> getDrugItems() async {
    // print("Getting drug items");
    var dbClient = await con.db;
    var res = await dbClient.query("drugs ORDER BY time DESC");

    // print("Got drug items $res");
    List<DrugItem> list =
        res.isNotEmpty
            ? res.map((c) => DrugItem.fromMap(c)).toList()
            : null;
    // print("Transformed to $list");
    return list;
  }



addDrug(DrugItem item) async {
    var dbClient = await con.db;
    var strdate = item.time.toIso8601String();
    var res = await dbClient.execute(
      "INSERT INTO drugs (name, time, img) "
          "VALUES (\"${item.name}\", \"$strdate\", \"${item.imagePath}\");"
    );

    return res;

  }
}
