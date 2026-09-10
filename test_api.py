import urllib.request
import urllib.parse
import time
import json

start = time.time()
try:
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="use_demo"\r\n\r\n'
        f'true\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="forecast_years"\r\n\r\n'
        f'5\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="epochs"\r\n\r\n'
        f'10\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="degradation"\r\n\r\n'
        f'0.0003\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="eq_type"\r\n\r\n'
        f'Motor\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="vyahh"\r\n\r\n'
        f'7.0\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="vxahh"\r\n\r\n'
        f'6.0\r\n'
        f'--{boundary}--\r\n'
    ).encode('utf-8')

    req = urllib.request.Request("http://localhost:8000/api/forecast", data=body, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Accept': 'application/json'
    }, method='POST')

    with urllib.request.urlopen(req) as res:
        print(f"Status Code: {res.status}")
        data = json.loads(res.read().decode('utf-8'))
        print(f"Success! keys: {data.keys()}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode('utf-8'))
except Exception as e:
    print(e)
print(f"Time taken: {time.time() - start:.2f} seconds")
