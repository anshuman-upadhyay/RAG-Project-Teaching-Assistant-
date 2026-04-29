# The segment section of each clip holds the details about the chunks(smallparts)
import os
import whisper
import json
model = whisper.load_model("large-v2")
filepath = "/home/anshuman/D_Drive/Coding/TeachingAssistant/ex-audio/3_realme-Buds-T500-Pro-Review-Bluetooth-6-_Media_g67kuqZyM_g_002_720p.mp4.mp3"
result = model.transcribe(audio = filepath,
                          language = "hi",
                          task = "translate")
# print(result["text"])
chunks = []
for segment in result["segments"]:
    chunks.append({ "start" : segment["start"],
                   "end" : segment["end"],
                   "text" : segment["text"], })
with open("output.json","w") as f:
    json.dump(chunks,f,indent=4)
    