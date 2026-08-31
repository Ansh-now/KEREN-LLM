#!/usr/bin/env python3
"""Repair legacy Gold records whose code-mode outputs were prose-only.

This script only replaces final_output.text for the known V0.1 code-quality
failures reported by validators/validate_dataset.py --all-gold. It does not
change user input, intent, safety, plan, decision, or labels.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "datasets" / "gold"

REPAIRS = {
"0222": '''```python
from pathlib import Path

def find_project(root: str, marker: str = "pubspec.yaml") -> Path:
    base = Path(root).expanduser().resolve()
    matches = list(base.rglob(marker))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one project, found {len(matches)}")
    return matches[0].parent

if __name__ == "__main__":
    print(find_project("."))
```''',
"0234": '''```python
import json
from pathlib import Path

def load_jsonl(path: str):
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {n}: {exc}") from exc

for row in load_jsonl("events.jsonl"):
    print(row)
```''',
"0243": '''```dart
import 'dart:convert';
import 'dart:io';

Future<void> main() async {
  final socket = await WebSocket.connect('ws://127.0.0.1:8000/ws');
  socket.listen((raw) {
    final data = jsonDecode(raw as String) as Map<String, dynamic>;
    stdout.writeln('event=${data['id']} type=${data['type']}');
  });
}
```''',
"0253": '''```python
from fastapi import FastAPI, HTTPException

app = FastAPI()
STATE: dict[str, str] = {}

@app.post("/tasks/{task_id}/cancel")
def cancel(task_id: str):
    state = STATE.get(task_id)
    if state is None:
        raise HTTPException(404, "task not found")
    if state in {"complete", "failed", "cancelled"}:
        return {"task_id": task_id, "state": state, "changed": False}
    STATE[task_id] = "cancel_requested"
    return {"task_id": task_id, "state": "cancel_requested", "changed": True}
```''',
"0266": '''```python
import sqlite3

def claim_task(db: sqlite3.Connection, worker_id: str):
    db.execute("BEGIN IMMEDIATE")
    row = db.execute("SELECT id FROM tasks WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
    if not row:
        db.rollback(); return None
    task_id = row[0]
    changed = db.execute("UPDATE tasks SET state='running', worker_id=? WHERE id=? AND state='queued'", (worker_id, task_id)).rowcount
    db.commit()
    return task_id if changed == 1 else None
```''',
"0271": '''```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';

class KerenSocket {
  KerenSocket(this.url);
  final String url;
  WebSocket? _ws;
  bool _closed = false;
  final _seen = <String>{};
  final _events = StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get events => _events.stream;

  Future<void> connect() async {
    var delay = const Duration(seconds: 1);
    while (!_closed) {
      try {
        _ws = await WebSocket.connect(url);
        await for (final raw in _ws!) {
          final e = jsonDecode(raw as String) as Map<String, dynamic>;
          final id = e['id']?.toString();
          if (id == null || _seen.add(id)) _events.add(e);
        }
      } catch (_) {}
      if (_closed) break;
      await Future.delayed(delay);
      if (delay < const Duration(seconds: 16)) delay *= 2;
    }
  }
  Future<void> dispose() async { _closed = true; await _ws?.close(); await _events.close(); }
}
```''',
"0272": '''```kotlin
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.flow.StateFlow

data class Device(val id:String, val name:String, val online:Boolean)
class DeviceVm(val devices: StateFlow<List<Device>>)

@Composable
fun DeviceList(vm: DeviceVm) {
    val devices = vm.devices.collectAsStateWithLifecycle().value
    LazyColumn {
        items(devices, key = { it.id }) { d ->
            Text(if (d.online) d.name else "${d.name} • offline")
        }
    }
}
```''',
"0273": '''```cpp
#include <WiFi.h>
#include <HTTPClient.h>

String eventId;
void sendWake(const String& id) {
  HTTPClient http;
  http.begin("http://192.168.1.2:8000/events");
  http.addHeader("Content-Type", "application/json");
  String body = "{\"event_id\":\"" + id + "\",\"type\":\"wake\"}";
  int code = http.POST(body);
  http.end();
  if (code < 200 || code >= 300) delay(500); // retry with SAME id
}
void setup(){ WiFi.begin("ssid","pass"); }
void loop(){
  if (digitalRead(34) == HIGH) {
    if (eventId.isEmpty()) eventId = String((uint32_t)esp_random(), HEX);
    sendWake(eventId);
    delay(1000);
  }
}
```''',
"0274": '''```python
import sqlite3
from fastapi import FastAPI

app = FastAPI(); DB="tasks.db"
@app.post('/workers/{worker_id}/claim')
def claim(worker_id: str):
    con=sqlite3.connect(DB, isolation_level=None)
    con.execute('BEGIN IMMEDIATE')
    row=con.execute("SELECT id FROM tasks WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
    if not row: con.execute('COMMIT'); return {'task':None}
    task_id=row[0]
    n=con.execute("UPDATE tasks SET state='running',worker_id=? WHERE id=? AND state='queued'",(worker_id,task_id)).rowcount
    con.execute('COMMIT')
    return {'task': task_id if n==1 else None}
```''',
"0281": '''```python
import json, sys

def journey(path: str, task_id: str):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            try: e=json.loads(line)
            except json.JSONDecodeError: continue
            if e.get('task_id')==task_id: rows.append(e)
    rows.sort(key=lambda e:(e.get('sequence',10**9), e.get('timestamp','')))
    for e in rows: print(e.get('timestamp'), e.get('source'), e.get('type'), e.get('state'))

if __name__=='__main__': journey(sys.argv[1], sys.argv[2])
```''',
"0282": '''```cpp
#include <AccelStepper.h>
AccelStepper x(AccelStepper::DRIVER,2,5);
const int LIMIT=9; enum HState{SEEK,BACKOFF,TOUCH,DONE}; HState s=SEEK;
void setup(){ pinMode(LIMIT,INPUT_PULLUP); x.setMaxSpeed(400); }
void loop(){
  if(s==SEEK){ x.setSpeed(-250); x.runSpeed(); if(!digitalRead(LIMIT)){x.move(40);s=BACKOFF;} }
  else if(s==BACKOFF){ x.run(); if(x.distanceToGo()==0)s=TOUCH; }
  else if(s==TOUCH){ x.setSpeed(-80); x.runSpeed(); if(!digitalRead(LIMIT)){x.setCurrentPosition(0);s=DONE;} }
}
```''',
"0291": '''```dart
import 'package:flutter/material.dart';

class TaskItem { final String id,state; const TaskItem(this.id,this.state); }
class TaskQueue extends StatelessWidget {
  const TaskQueue({super.key, required this.tasks});
  final List<TaskItem> tasks;
  @override Widget build(BuildContext context) => ListView.builder(
    itemCount: tasks.length,
    itemBuilder: (_,i){ final t=tasks[i]; return ListTile(key:ValueKey(t.id),title:Text(t.id),subtitle:Text(t.state)); },
  );
}
```''',
"0292": '''```kotlin
import kotlinx.coroutines.*
import java.net.HttpURLConnection
import java.net.URL

class HeartbeatService(private val endpoint:String, private val nodeId:String) {
    private val scope=CoroutineScope(SupervisorJob()+Dispatchers.IO)
    fun start()=scope.launch {
        while(isActive){
            val c=URL(endpoint).openConnection() as HttpURLConnection
            c.requestMethod="POST"; c.doOutput=true
            c.outputStream.use{ it.write("{\"node_id\":\"$nodeId\"}".toByteArray()) }
            c.inputStream.close(); delay(10_000)
        }
    }
    fun stop()=scope.cancel()
}
```''',
"0293": '''```cpp
#include <Preferences.h>
Preferences prefs;
uint32_t seqNo=0;
void setup(){ prefs.begin("keren",false); seqNo=prefs.getUInt("seq",0); }
String nextEvent(){
  seqNo++; prefs.putUInt("seq",seqNo);
  return String("esp32-")+String(seqNo);
}
void loop(){
  // Persist monotonic event id before enqueue/send so retries reuse identity.
  static String pending;
  if(pending.isEmpty()) pending=nextEvent();
}
```''',
"0294": '''```python
from fastapi import FastAPI, HTTPException
app=FastAPI(); tasks={}
@app.post('/tasks/{tid}/cancel')
def cancel(tid:str):
    t=tasks.get(tid)
    if t is None: raise HTTPException(404,'task not found')
    if t['state'] in {'complete','failed','cancelled'}: return t
    t['cancel_requested']=True
    return {'id':tid,'state':t['state'],'cancel_requested':True}
```''',
"0301": '''```cpp
const int RELAY=23;
unsigned long lastHeartbeat=0;
void setSafe(){ digitalWrite(RELAY,LOW); }
void setup(){ pinMode(RELAY,OUTPUT); setSafe(); }
void loop(){
  if(millis()-lastHeartbeat>5000) setSafe();
  if(Serial.available()){
    String cmd=Serial.readStringUntil('\n');
    lastHeartbeat=millis();
    if(cmd=="RELAY_ON") digitalWrite(RELAY,HIGH);
    else if(cmd=="RELAY_OFF") setSafe();
  }
}
```''',
"0311": '''```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class CoreApi {
  CoreApi(this.base,this.token); final String base,token;
  Future<Map<String,dynamic>> getTask(String id) async {
    for(var n=0;n<3;n++){
      try{
        final r=await http.get(Uri.parse('$base/tasks/$id'),headers:{'Authorization':'Bearer $token'}).timeout(const Duration(seconds:5));
        if(r.statusCode==200) return jsonDecode(r.body);
        if(r.statusCode<500) throw Exception('HTTP ${r.statusCode}');
      }catch(e){ if(n==2) rethrow; }
      await Future.delayed(Duration(milliseconds:300*(n+1)));
    }
    throw StateError('unreachable');
  }
}
```''',
"0312": '''```kotlin
sealed interface UiState<out T>{ data object Loading:UiState<Nothing>; data class Data<T>(val value:T):UiState<T>; data class Error(val message:String):UiState<Nothing> }

fun render(state: UiState<List<String>>): String = when(state){
    UiState.Loading -> "Loading"
    is UiState.Data -> state.value.joinToString()
    is UiState.Error -> "Error: ${state.message}"
}
```''',
"0313": '''```cpp
#include <WiFi.h>
unsigned long nextTry=0; uint32_t backoff=1000;
void setup(){ WiFi.mode(WIFI_STA); }
void loop(){
  if(WiFi.status()==WL_CONNECTED){ backoff=1000; return; }
  if(millis()>=nextTry){
    WiFi.begin("ssid","pass");
    nextTry=millis()+backoff;
    backoff=min<uint32_t>(backoff*2,30000);
  }
}
```''',
"0314": '''```python
import asyncio, httpx
async def worker(base:str,node:str):
    async with httpx.AsyncClient(timeout=5) as c:
        while True:
            await c.post(f'{base}/nodes/{node}/heartbeat')
            r=await c.post(f'{base}/workers/{node}/claim')
            task=r.json().get('task')
            if task:
                if task.get('cancel_requested'): await c.post(f"{base}/tasks/{task['id']}/cancelled")
                else: await c.post(f"{base}/tasks/{task['id']}/ack")
            await asyncio.sleep(1)
```''',
"0315": '''```cpp
class MovingAverage {
  float buf[60]{}; int i=0,n=0; float sum=0;
public:
  float add(float x){ sum-=buf[i]; buf[i]=x; sum+=x; i=(i+1)%60; if(n<60)n++; return sum/n; }
};
MovingAverage avg;
void setup(){ Serial.begin(115200); }
void loop(){ Serial.println(avg.add(analogRead(34))); delay(100); }
```''',
"0321": '''```python
from pathlib import Path
import subprocess

def create_pr_branch(issue:int):
    branch=f"fix/issue-{issue}"
    subprocess.run(["git","switch","-c",branch],check=True)
    Path(".keren_issue").write_text(str(issue))
    subprocess.run(["git","add",".keren_issue"],check=True)
    subprocess.run(["git","commit","-m",f"Start issue #{issue}"],check=True)
    return branch
```''',
"0331": '''```dart
import 'dart:convert';
class JsonlDecoder {
  String _buf='';
  Iterable<Map<String,dynamic>> add(String chunk) sync* {
    _buf+=chunk; final parts=_buf.split('\n'); _buf=parts.removeLast();
    for(final line in parts){ if(line.trim().isNotEmpty) yield jsonDecode(line) as Map<String,dynamic>; }
  }
}
```''',
"0332": '''```kotlin
fun blePermissions(sdk:Int): Array<String> = if(sdk>=31) arrayOf(
    android.Manifest.permission.BLUETOOTH_SCAN,
    android.Manifest.permission.BLUETOOTH_CONNECT
) else arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION)
```''',
"0333": '''```python
import time
from fastapi import FastAPI, WebSocket
app=FastAPI(); nodes={}
@app.websocket('/ws/nodes/{node_id}')
async def ws(ws:WebSocket,node_id:str):
    await ws.accept(); nodes[node_id]={'ws':ws,'seen':time.time()}
    try:
        while True: await ws.receive_text(); nodes[node_id]['seen']=time.time()
    finally: nodes.pop(node_id,None)
def stale(max_age=30): return [n for n,v in nodes.items() if time.time()-v['seen']>max_age]
```''',
"0334": '''```scad
$fn=48;
board=[52,28,8]; wall=2; gap=0.6;
difference(){
  cube([board[0]+2*(wall+gap),board[1]+2*(wall+gap),board[2]+wall]);
  translate([wall,wall,wall]) cube([board[0]+2*gap,board[1]+2*gap,board[2]+2]);
  translate([-1,10,4]) cube([wall+3,8,4]);
}
```''',
"0335": '''```python
import FreeCAD as App, Part
box=Part.makeBox(56,32,12)
cavity=Part.makeBox(52,28,10,App.Vector(2,2,2))
shape=box.cut(cavity)
doc=App.newDocument('Enclosure'); obj=doc.addObject('Part::Feature','Body'); obj.Shape=shape
doc.recompute(); Part.export([obj],'esp32_enclosure.stl')
```''',
"0341": '''```cpp
#include <Preferences.h>
Preferences p;
void markBootOk(){ p.begin("ota",false); p.putBool("pending",false); p.end(); }
void setup(){
  p.begin("ota",false); bool pending=p.getBool("pending",false); p.end();
  // New image must call markBootOk() only after network/core health succeeds.
  if(!pending) markBootOk();
}
void loop(){}
```''',
"0351": '''```dart
import 'dart:async'; import 'dart:io';
class ReconnectingWs{ final Uri uri; WebSocket? _ws; bool stop=false;
ReconnectingWs(this.uri);
Future<void> run() async{ var d=1; while(!stop){ try{ _ws=await WebSocket.connect(uri.toString()); d=1; await _ws!.done; }catch(_){} if(stop)break; await Future.delayed(Duration(seconds:d)); d=(d*2).clamp(1,16); }}
Future<void> close() async{stop=true; await _ws?.close();}}
```''',
"0352": '''```kotlin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

data class NodeStatus(val id:String,val online:Boolean)
class NodesVm{
 private val _nodes=MutableStateFlow<List<NodeStatus>>(emptyList())
 val nodes:StateFlow<List<NodeStatus>>=_nodes
 fun update(v:List<NodeStatus>){_nodes.value=v}
}
```''',
"0353": '''```cpp
#include <ArduinoJson.h>
const int RELAY=23, PWM=25;
void setup(){Serial.begin(115200);pinMode(RELAY,OUTPUT);ledcAttach(PWM,5000,8);}
void loop(){ if(!Serial.available())return; StaticJsonDocument<128>d; if(deserializeJson(d,Serial))return; String c=d["cmd"]|""; if(c=="relay")digitalWrite(RELAY,d["on"]?HIGH:LOW); else if(c=="pwm")ledcWrite(PWM,constrain((int)d["value"],0,255)); }
```''',
"0354": '''```python
import sqlite3, uuid
from fastapi import FastAPI
app=FastAPI(); con=sqlite3.connect('keren.db',check_same_thread=False)
con.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,key TEXT UNIQUE,payload TEXT)')
@app.post('/tasks')
def create(payload:str,key:str):
 r=con.execute('SELECT id FROM tasks WHERE key=?',(key,)).fetchone()
 if r:return {'id':r[0],'created':False}
 tid=str(uuid.uuid4()); con.execute('INSERT INTO tasks VALUES(?,?,?)',(tid,key,payload)); con.commit(); return {'id':tid,'created':True}
```''',
"0355": '''```python
from dataclasses import dataclass
@dataclass
class Node: name:str; capabilities:set[str]; load:float; online:bool=True
def choose(nodes:list[Node], required:set[str]):
    eligible=[n for n in nodes if n.online and required<=n.capabilities]
    if not eligible: raise RuntimeError('no eligible node')
    return min(eligible,key=lambda n:n.load)
```''',
"0356": '''```dart
import 'package:flutter/material.dart';
class SafeButton extends StatelessWidget{ const SafeButton({super.key,required this.label,required this.onPressed}); final String label; final VoidCallback onPressed;
@override Widget build(BuildContext c)=>Semantics(button:true,label:label,enabled:true,child:ElevatedButton(onPressed:onPressed,child:Text(label)));}
```''',
"0357": '''```kotlin
class EventDeduper(private val max:Int=1024){
 private val seen=LinkedHashSet<String>()
 @Synchronized fun accept(id:String):Boolean{
   if(!seen.add(id)) return false
   if(seen.size>max) seen.remove(seen.first())
   return true
 }
}
```''',
"0358": '''```cpp
template<size_t N> class Ring{ float b[N]{}; size_t i=0,n=0; public: void push(float v){b[i]=v;i=(i+1)%N;if(n<N)n++;} float at(size_t k)const{return b[k%N];} size_t size()const{return n;} };
Ring<60> samples;
void setup(){Serial.begin(115200);} void loop(){samples.push(analogRead(34));delay(100);}
```''',
"0359": '''```python
import time
from fastapi import FastAPI
app=FastAPI(); heartbeats={}
@app.post('/nodes/{node}/heartbeat')
def hb(node:str): heartbeats[node]=time.time(); return {'ok':True}
@app.get('/nodes')
def nodes(max_age:float=30):
 now=time.time(); return [{'id':n,'online':now-t<=max_age,'age':now-t} for n,t in heartbeats.items()]
```''',
"0360": '''```python
import time

def retry(fn, attempts=4, base=0.25, retryable=(TimeoutError,ConnectionError)):
    for i in range(attempts):
        try: return fn()
        except retryable:
            if i==attempts-1: raise
            time.sleep(base*(2**i))
```
''',
}


def main() -> None:
    found=set(); changed_files=0; changed_records=0
    for path in sorted(GOLD.glob('gold_v0.1_*.jsonl')):
        rows=[]; changed=False
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip(): continue
            rec=json.loads(raw); rid=str(rec.get('id'))
            if rid in REPAIRS:
                rec.setdefault('final_output',{})['mode']='code'
                rec['final_output']['text']=REPAIRS[rid]
                rec.setdefault('task',{})['output_mode']='code'
                rec['task']['task_type']='code'
                found.add(rid); changed=True; changed_records+=1
            rows.append(json.dumps(rec,ensure_ascii=False,separators=(',',':')))
        if changed:
            path.write_text('\n'.join(rows)+'\n',encoding='utf-8'); changed_files+=1
    missing=sorted(set(REPAIRS)-found)
    print(f'Repaired code Gold: files={changed_files}, records={changed_records}, missing={len(missing)}')
    if missing: print('Missing IDs:', ', '.join(missing))

if __name__=='__main__': main()
