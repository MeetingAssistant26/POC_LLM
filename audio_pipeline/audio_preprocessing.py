import os
from pydub import AudioSegment

input_folder = "audio"
output_folder = "audio_clean"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    if file.endswith((".wav", ".mp3", ".m4a")):

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.split(".")[0] + ".wav")

        print(f"Processing {file}")

        audio = AudioSegment.from_file(input_path)

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        audio.export(output_path, format="wav")

print("Audio preprocessing completed.")