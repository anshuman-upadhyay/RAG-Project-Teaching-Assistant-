import whisper
model = whisper.load_model("base")
result = model.transcribe(audio = "ex-audio/8_Forms-and-input-tags-in-HTML-.mp3",
                          language = "hi",
                          task = "translate")
print(result["text"])