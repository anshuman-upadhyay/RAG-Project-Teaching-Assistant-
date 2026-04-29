# The segment section of each clip holds the details about the chunks(smallparts)
import os
import whisper
import json
model = whisper.load_model("large-v2")
audios = os.listdir("ex-audio")
for audio in audios:
    number= audio.split("_")[0]
    title = audio.split("_")[1][:-20]
    print(number,title)
    result = model.transcribe(audio = f"ex-audio/{audio}",
                          language = "hi",
                          task = "translate",
                          word_timestamps= False)
    chunks = []
    for seg in result["segments"]:
        chunks.append({
                "number" : number,
                "title" : title,
                "start" : seg["start"],
                "end" : seg["end"],
                "text" : seg["text"]
            })
    chunks_with_metadata = {"chunks" : chunks,
                            "text" : result["text"]}
    with open(f"Json/{audio}.json","w") as f:
        json.dump(chunks_with_metadata,f,indent=4)
