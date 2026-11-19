from pyngrok import ngrok

ngrok.set_auth_token("38spu4omsKhYqZzckys7dknLJjW_b8pJsUb49SDPVpzbFmok")

url = ngrok.connect(8501)
print(f"Lien public disponible ici : {url.public_url}")

import time
while True:
    time.sleep(1)