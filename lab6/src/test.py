import requests

response = requests.get("https://qrng.anu.edu.au/API/jsonI.php?length=100&type=uint8")
quantum_random_numbers = response.json()["data"]
print(quantum_random_numbers)
