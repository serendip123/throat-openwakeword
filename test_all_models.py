# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import os
import glob
import colorama
import sys

# Initialize colorama for Windows ANSI support
colorama.init()

# Set stdout to UTF-8 to handle Japanese characters on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default="onnx",
    required=False,
)
parser.add_argument(
    "--list_devices",
    help="List all available input devices and exit",
    action="store_true",
    required=False,
)
parser.add_argument(
    "--device_index",
    help="The index of the input device to use",
    type=int,
    default=None,
    required=False,
)

args = parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()

# List devices if requested
if args.list_devices:
    print("Available Input Devices:")
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if int(dev["maxInputChannels"]) > 0:
            print(f"Index {i}: {dev['name']}")
    exit(0)

try:
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=args.device_index,
    )
except OSError as e:
    print(f"Error opening audio stream: {e}")
    print("Please check your microphone permissions and settings.")
    print("If you have multiple devices, try using --list_devices and --device_index.")
    exit(1)

# Find all onnx models in the current directory
model_paths = glob.glob("*.onnx")

if not model_paths:
    print("Error: No models found in current directory.")
    exit(1)

print(f"Loading {len(model_paths)} models...")
for path in model_paths:
    print(f" - {os.path.basename(path)}")

# Load pre-trained openwakeword models
owwModel = Model(
    wakeword_models=model_paths, inference_framework=args.inference_framework
)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#" * 100)
    print(f"Listening for wakewords using {n_models} models...")
    print("#" * 100)
    print("\n" * (n_models * 3))

    while True:
        # Get audio
        try:
            audio_data = mic_stream.read(CHUNK, exception_on_overflow=False)
        except OSError as e:
            print(f"Warning: Audio overflow or error: {e}")
            continue

        audio = np.frombuffer(audio_data, dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 20
        output_string_header = """
            Model Name             | Score | Wakeword Status
            ------------------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], ".20f").replace("-", "")

            status_text = "--" + " " * 20
            if scores[-1] > 0.5:
                status_text = "\033[1;32mWAKEWORD DETECTED!\033[0m"  # Green bold text

            # Truncate model name for display if too long, or pad
            display_name = mdl[:n_spaces]

            output_string_header += f"""{display_name}{" " * (n_spaces - len(display_name))}   | {curr_score[0:5]} | {status_text}
            """

        # Print results table
        print("\033[F" * (4 * n_models + 1))
        print(output_string_header, "                             ", end="\r")
