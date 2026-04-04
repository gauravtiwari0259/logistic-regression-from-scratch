import requests
import requests
c = input(str("enter"))
l="%23"+c

url = f"https://api.clashroyale.com/v1/players/{l}"

headers = {
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjYzYzk5NGFkLWY3NjUtNDNkZC1iZjk1LTljMjdhYWMxMzkyMyIsImlhdCI6MTc3MDgyNTUwNCwic3ViIjoiZGV2ZWxvcGVyL2IwNWE3YTY3LTEzNTgtY2JhNi0zZTlkLTYxZWEwNzNmMjU5ZSIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMTcuMjEzLjIwMC4zIiwiMjIwLjE1OC4xODMuMTYiLCIxMTIuMTMzLjIyMC4xMzYiXSwidHlwZSI6ImNsaWVudCJ9XX0.CU0iAyu-KAGZNeMICn64NptrhWSgf5ibGKijHyOJ1RQ0GGUnBtI8s-TtOJA9ZbBdu3I3efWCr1H4of2oGLkEyw"
}
response = requests.get(url, headers=headers)

#print(response.status_code)
#print(response.json())
g=response.json()
print("the tag is",g["tag"],g["name"],g["trophies"])

