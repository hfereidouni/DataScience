import requests

r = requests.post(
	"http://localhost:8080/predict", 
	json={'qq':"pp","qwdqwd":123}
)

print(r.json())