import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [0.5, 1.2, 3.4, 4.5, 0.6, 0.7, 1.1, 2.2, 3.1, 4.3, 5.0, 1.5, 0.9, 2.4]}
response = requests.post(url, json=data)
print("Response:", response.json())

