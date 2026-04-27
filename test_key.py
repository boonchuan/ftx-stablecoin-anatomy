import os
import requests

# Find the flows script
flows_file = [f for f in os.listdir('.') if 'flows' in f and f.endswith('.py')][0]
print("Reading:", flows_file)

# Extract the key
content = open(flows_file).read()
key = content.split('ETHERSCAN_API_KEY = "')[1].split('"')[0]
print("Key length:", len(key))
print("Key first 4 chars:", key[:4])
print("Key last 4 chars:", key[-4:])

# Test against simplest Etherscan endpoint
r = requests.get(
    'https://api.etherscan.io/api',
    params={'module': 'stats', 'action': 'ethsupply', 'apikey': key}
)
print("Response:", r.json())