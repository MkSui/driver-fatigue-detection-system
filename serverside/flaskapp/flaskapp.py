from http.client import BAD_GATEWAY, NOT_FOUND
from os import abort
from flask import Flask, request, jsonify
import json
from datetime import datetime, date, timedelta  # timedelta 修改时间函数 日期
import time
import hashlib
from gevent import pywsgi
import gps
TimeList=[]
recdic={}

if __name__ == '__main__':
	app = Flask(__name__)

	@app.route('/touroku')
	def touroku():
		loginkey = request.args.get('loginkey')
		key=time.strftime("%m-%d-%H", time.localtime())
		hash5 = hashlib.md5()
		hash5.update(key.encode("utf-8"))
		if loginkey == "114514" or loginkey == (hash5.hexdigest()).upper():
			global recdic
			try:
				jsondata = request.args.get('jsondata')
				recdic = json.loads(jsondata)
				recdic["longitude"],recdic["latitude"] = gps.wgs84_to_gcj02(recdic["longitude"],recdic["latitude"])

			except Exception as e:
					print(e)
					return "BAD ARGS"
			print(TimeList)
			#if loginkey == (hash5.hexdigest()).upper():
			if recdic['LastTime'] not in TimeList:
				TimeList.append(recdic['LastTime'])
			return "Verified"
		else:
			return """<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
"""
	@app.route('/morau')
	def morau():
		loginkey = request.args.get('loginkey')
		global recdic
		key=time.strftime("%m-%d-%H", time.localtime())
		hash5 = hashlib.md5()
		hash5.update(key.encode("utf-8"))
		if loginkey == "114514" or loginkey == (hash5.hexdigest()).upper():
			
			json_string = json.dumps(recdic)
			return json_string.encode('utf-8')
		else:
			return """<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
"""
	@app.route('/seiketsu')
	def seiketsu():
		loginkey = request.args.get('loginkey')
		global recdic,TimeList
		key=time.strftime("%m-%d-%H", time.localtime())
		hash5 = hashlib.md5()
		hash5.update(key.encode("utf-8"))
		if loginkey == "114514" or loginkey == (hash5.hexdigest()).upper():
				TimeList=[]
				recdic={}
		else:
			return """<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
"""

	#app.run()
	server = pywsgi.WSGIServer(('127.0.0.1', 11451), app)
	server.serve_forever()
