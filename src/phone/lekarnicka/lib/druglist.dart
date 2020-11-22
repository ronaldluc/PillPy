import 'dart:async';
import 'package:flutter/material.dart';
import 'package:lekarnicka/bigbutton.dart';
import 'package:lekarnicka/drugitem.dart';

import 'querry.dart';
import 'drugelement.dart';

class DrugList extends StatefulWidget {
  final QueryCtr _query;
  DrugList(this._query);

  @override
  State<StatefulWidget> createState() {
    return new DrugListState();
  }
}

class DrugListState extends State<DrugList> {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Lékárnička ‒ Seznam léků')),
      body: Column(
        children: [
          Container(
            // alignment: Alignment.topCenter,
            padding: EdgeInsets.all(10),
            child: BigButton(
              "přidat lék",
              () => Navigator.pop(context),
            ),
          ),
          Expanded(
            child: FutureBuilder<List>(
              future: widget._query.getDrugItems(),
              initialData: List(),
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  print(snapshot.error);
                }
                return snapshot.hasData
                    ? ListView.separated(
                        padding: const EdgeInsets.all(10.0),
                        itemCount: snapshot.data.length,
                        itemBuilder: (context, i) {
                          return DrugElement(snapshot.data[i]);
                        },
                        separatorBuilder: (context, index) => Divider(
                          color: Colors.black,
                        ),
                      )
                    : Center(
                        child: CircularProgressIndicator(),
                      );
              },
            ),
          ),
        ],
      ),
    );
  }
}
