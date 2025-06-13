import http.client
import json
from utils import load_image_as_base64, init_logger
image_path = "/Users/aonian/Documents/硕士/科研/多模态检测/论文实验/mac-agiqa/image/000-2.png"
image_base64 = load_image_as_base64(image_path)
conn = http.client.HTTPSConnection("kfcv50.link")
YOUR_API_KEY = 'sk-CC3rA9d90JLLkUFQ42C04a21F8074d6e9d93C511E4D8Cc6d'
payload = json.dumps({
   "model": "gpt-4-vision-preview",
   "stream": False,
   "messages": [
      {
         "role": "user",
         "content": [
            {
               "type": "text",
               "text": "这张图片有什么"
            },
            {
               "type": "image_url",
               "image_url": {
                  "url": "https://github.com/dianping/cat/raw/master/cat-home/src/main/webapp/images/logo/cat_logo03.png"
               }
            }
         ]
      }
   ],
   "max_tokens": 400
})
headers = {
   'Accept': 'application/json',
   'Authorization': 'sk-CC3rA9d90JLLkUFQ42C04a21F8074d6e9d93C511E4D8Cc6d',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))