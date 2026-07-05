import urllib.request

resp = urllib.request.urlopen('http://127.0.0.1:8000/')
print('STATUS', resp.getcode())
print('HEADERS', resp.getheaders())
print()
print(resp.read().decode('utf-8'))
