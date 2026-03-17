import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

input_folder = "audio"
output_folder = "audio_clean"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    if file.endswith((".wav", ".mp3", ".m4a")):

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.split(".")[0] + ".wav")

        print(f"Processing {file}")

        audio = AudioSegment.from_file(input_path)

        # Convert to mono
        audio = audio.set_channels(1)

        # Convert sample rate
        audio = audio.set_frame_rate(16000)

        # Normalize volume
        change_in_dBFS = -20.0 - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # Remove long silence
        chunks = split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=audio.dBFS - 14,
            keep_silence=300
        )

        processed_audio = AudioSegment.empty()

        for chunk in chunks:
            processed_audio += chunk

        processed_audio.export(output_path, format="wav")

print("Audio preprocessing completed.")