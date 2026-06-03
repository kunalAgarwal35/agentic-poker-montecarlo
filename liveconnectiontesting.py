import requests
import json

# The base URL for your Flask application
# base_url = 'http://127.0.0.1:5000'
base_url = "http://pptloadbalancer-1427274483.ap-south-1.elb.amazonaws.com"


def update_holecards(tableid, heroname, holecards):
    url = f"{base_url}/update_holecards"
    payload = {
        "tableid": tableid,
        "heroname": heroname,
        "holecards": holecards
    }
    response = requests.post(url, json=payload)
    print("Update Holecards Response:", response.text)

def update_community_cards(heroname, communitycards):
    url = f"{base_url}/update_community_cards"
    payload = {
        "heroname": heroname,
        "communitycards": communitycards,
    }
    response = requests.post(url, json=payload)
    print("Update Community Cards Response:", response.text)

def draw_table():
    url = f"{base_url}/draw_table"
    response = requests.get(url)
    print("Draw Table Response:", response.text)

def clear_all():
    url = f"{base_url}/clear_all"
    response = requests.post(url)
    print("Clear All Response:", response.text)

# Test the API
import time
update_holecards('1', 'HeroName4', 'AsKdTcJc')
time.sleep(5)
update_community_cards('HeroName1', '2cjc8h9s')
draw_table()
clear_all()
