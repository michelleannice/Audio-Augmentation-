import argparse
import os
import re
import time
import numpy as np
import pyroomacoustics
import glob
import parselmouth
from pydub import AudioSegment
from pydub.playback import play
from parselmouth.praat import call
from scipy.io import wavfile
from operator import itemgetter
import wave
import multiprocessing
import random
import librosa

def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time

# from typing import List, Dict
# from specAugment import spec_augment_tensorflow
import audio
from audio import AudioItem

ORIGINAL_WAV_MAPPING_PATH = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/data/wav.scp"
ORIGINAL_UTT2SPK_FILE = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/data/utt2spk"
ORIGINAL_TEXT_FILE = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/data/text"
ORIGINAL_WAV_PATH_PREFIX = "/data/data_kaldi/Data_Training_2021/Wav_resampled/"
ORIGINAL_CORPUSTXT_FILE = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/data/corpus.txt"
ORIGINAL_WAV_FILE = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav"

RESULT_PATH = "data_spec_augment"
RESULT_UTT2SPK_FILE = os.path.join(RESULT_PATH, "utt2spk")

DEFAULT_INPUT_DATA_DIR = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data"
DEFAULT_OUTPUT_DATA_DIR = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented txt"
DEFAULT_OUTPUT_WAV_DIR = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav"

# def open_utt2spk(filepath: str):
def open_utt2spk(filepath):
    utt2spk = {}

    with open(filepath, "r") as spk2utt_file:
        lines = spk2utt_file.readlines()
        for line in lines:
            audio_id, speaker = line.split()
            # utt2spk[audio_id] = line_split[0]
            # utt2spk[speaker] = re.sub(pattern=r'\n', repl='', string=line_split[1])
            utt2spk[audio_id] = speaker
            # utt2spk.update({audio_id: speaker})
    return utt2spk

# def open_text
def open_text(filepath):
    daftar_teks = {}

    with open(filepath, "r") as text_file:
        lines = text_file.readlines()
        for line in lines:
            audio_id, text = line.split(" ",1)
            # audio_id = line_split[0]
            # text = re.sub(pattern=r'\n', repl='', string=line_split[1])
            daftar_teks [audio_id] = text
            # daftar_teks.update({audio_id: text})
    return daftar_teks

def open_wavscp(filepath):
    wavscp = {}

    with open(filepath, "r") as wavscp_file:
        lines = wavscp_file.readlines()
        for line in lines:
            audio_id, audio_path = line.split(" ")
            wavscp[audio_id] = audio_path
    return wavscp

# def create_audio_item_list(filepath: str, utt2spk: Dict) -> List[AudioItem]:
def create_audio_item_list(filepath, wavscp_dict, utt2spk_dict, daftar_teks_dict):
    # utt2spk_dict = {}
    # daftar_teks_dict = {}
    list_audio_items = list()
    with open(filepath, "r") as wav_mapping_file:
        lines = wav_mapping_file.readlines()
        limit = 10
        counter = 0
        for line in lines:
            line_split = line.split(" ")
            audio_id = line_split[0]
            audio_path = wavscp_dict.get(audio_id)
            speaker = utt2spk_dict.get(audio_id)
            text = daftar_teks_dict.get(audio_id)
            list_audio_items.append(AudioItem(audio_id, audio_path, speaker, text))
            counter = counter + 1
            if counter == limit:
                break
        return list_audio_items

# def augment_sound(original_audio_item: AudioItem) -> AudioItem:
def combine_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_combine_audio"
    generated_path = original_audio_item.audio_path[:-5]+ "_combine_audio"+ original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def pitch_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_pitch_change"
    generated_path = original_audio_item.audio_path[:-5] + "_pitch_change" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def time_stretch_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_time_stretch"
    generated_path = original_audio_item.audio_path[:-5] + "_time_stretch" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def reverb_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_reverb"
    generated_path = original_audio_item.audio_path[:-5] + "_reverb" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def combine_pitch_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_combine_pitch"
    generated_path = original_audio_item.audio_path[:-5] + "_combine_pitch" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def combine_time_stretch_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_combine_time_stretch"
    generated_path = original_audio_item.audio_path[:-5] + "_combine_time_stretch" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)

def combine_reverb_augment_sound(original_audio_item):
    generated_audio_id = original_audio_item.audio_id + "_combine_reverb"
    generated_path = original_audio_item.audio_path[:-5] + "_combine_reverb" + original_audio_item.audio_path[-5:]
    generated_speaker = original_audio_item.speaker
    generated_text = original_audio_item.text
    return AudioItem(audio_id=generated_audio_id, audio_path=generated_path, speaker=generated_speaker, text=generated_text)


def room_simulation(filepath):
    file_name = os.path.basename(filepath)

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.5  # seconds
    room_dim = [6, 6, 3]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio = wavfile.read(filepath)

    e_absorption, max_order = pyroomacoustics.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pyroomacoustics.ShoeBox(
        room_dim, fs=fs, materials=pyroomacoustics.Material(e_absorption), max_order=max_order
    )

    # place the source in the room
    room.add_source([2.5, 3, 1.76], signal=audio, delay=0.5)

    # # define the locations of the microphones
    mic_locs = np.c_[
        [5, 1.25, 1.25], [2, 1.25, 1.25],# mic 1  # mic 2
    ]

    # # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav reverb/{file_name[:-4]}_reverb.wav",
        norm=True,
        bitdepth=np.int16,
    )

def combine_multiple_audio(filepath):
    file_name = os.path.basename(filepath)
    audio1 = AudioSegment.from_file(filepath)  # your first audio file
    audio2 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/footsteps-4.wav")  # your second audio file
    audio3 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/musan/music/fma-western-art/music-fma-wa-0023.wav")  # your third audio file

    audio2_volume = audio2 - 7
    audio3_volume = audio3 - 20

    mixed = audio1.overlay(audio2_volume)  # combine , superimpose audio files
    mixed1 = mixed.overlay(audio3_volume)  # Further combine , superimpose audio files
    # If you need to save mixed file
    mixed1.export(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav combine/{file_name[:-4]}_combine_audio.wav", format='wav')  # export mixed  audio file
    # mixed1.to_wav(file_name)

def combine_hospital_audio(filepath):
    file_name = os.path.basename(filepath)
    noises = []
    augmented_noise = []

    i = 0

    for filename in glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/Hospital Noise/**/*.wav", recursive=True):
        noises.append(filename)

    while i < len(noises):
        audio1 = AudioSegment.from_file(filepath)  # your first audio file
        audio2 = AudioSegment.from_file(noises[i])

        audio2_volume = audio2 - 20
        file_name_noise = os.path.basename(noises[i])
        mixed = audio1.overlay(audio2_volume)  # combine , superimpose audio files
        mixed.export(
            f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav combine hospital noise/{file_name[:-4]}_{file_name_noise[:-4]}_combine_hospital_noise.wav",
            format='wav')  # export mixed  audio file
        augmented_noise.append(mixed)
        i += 1

def change_pitch_parselmouth(filepath):
    file_name = os.path.basename(filepath)

    if file_name[0] == "F":
        sound = parselmouth.Sound(filepath)
        manipulation = call(sound, "To Manipulation", 0.01, 30, 90)
        pitch_tier = call(manipulation, "Extract pitch tier")

        call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)

        call([pitch_tier, manipulation], "Replace pitch tier")
        sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")

        sound_octave_up.save(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav pitch change/{file_name[:-4]}_pitch_change.wav","WAV")

    else:
        sound = parselmouth.Sound(filepath)
        manipulation = call(sound, "To Manipulation", 0.01, 100, 200)
        pitch_tier = call(manipulation, "Extract pitch tier")

        call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)

        call([pitch_tier, manipulation], "Replace pitch tier")
        sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")

        sound_octave_up.save(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav pitch change/{file_name[:-4]}_pitch_change.wav","WAV")

def time_stretch(filepath):
    file_name = os.path.basename(filepath)
    CHANNELS = 1
    swidth = 2
    Change_RATE = 0.85

    spf = wave.open(filepath, 'rb')
    RATE = spf.getframerate()
    signal = spf.readframes(-1)

    wf = wave.open(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav time stretch/{file_name[:-4]}_time_stretch.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(swidth)
    wf.setframerate(RATE * Change_RATE)
    wf.writeframes(signal)
    wf.close()

def combine_pitch(filepath):
    file_name = os.path.basename(filepath)

    audio1 = AudioSegment.from_file(filepath)  # your first audio file
    audio2 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/footsteps-4.wav")  # your second audio file
    audio3 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/musan/music/fma-western-art/music-fma-wa-0023.wav")  # your third audio file

    audio2_volume = audio2 - 7
    audio3_volume = audio3 - 20

    mixed = audio1.overlay(audio2_volume)  # combine , superimpose audio files
    mixed1 = mixed.overlay(audio3_volume)  # Further combine , superimpose audio files
    # If you need to save mixed file
    mixed1.export(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav combine pitch/{file_name[:-18]}_combine_pitch.wav",format='wav')  # export mixed  audio file
    # mixed1.to_wav(file_name)

def combine_time_stretch(filepath):
    file_name = os.path.basename(filepath)

    audio1 = AudioSegment.from_file(filepath)  # your first audio file
    audio2 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/footsteps-4.wav")  # your second audio file
    audio3 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/musan/music/fma-western-art/music-fma-wa-0023.wav")  # your third audio file

    audio2_volume = audio2 - 7
    audio3_volume = audio3 - 20

    mixed = audio1.overlay(audio2_volume)  # combine , superimpose audio files
    mixed1 = mixed.overlay(audio3_volume)  # Further combine , superimpose audio files
    # If you need to save mixed file
    mixed1.export(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav combine time stretch/{file_name[:-18]}_combine_time_stretch.wav",format='wav')  # export mixed  audio file
    # mixed1.to_wav(file_name)

def combine_reverb(filepath):
    file_name = os.path.basename(filepath)

    audio1 = AudioSegment.from_file(filepath)  # your first audio file
    audio2 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/footsteps-4.wav")  # your second audio file
    audio3 = AudioSegment.from_file("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/sample/musan/music/fma-western-art/music-fma-wa-0023.wav")  # your third audio file

    audio2_volume = audio2 - 7
    audio3_volume = audio3 - 20

    mixed = audio1.overlay(audio2_volume)  # combine , superimpose audio files
    mixed1 = mixed.overlay(audio3_volume)  # Further combine , superimpose audio files
    # If you need to save mixed file
    mixed1.export(f"/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav combine reverb/{file_name[:-18]}_combine_reverb.wav",format='wav')  # export mixed  audio file
    # mixed1.to_wav(file_name)

if __name__ == '__main__':
    start_time = time.time()

    # all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav",
    #                           recursive=True)
    # all_filenames_combine_pitch = glob.glob(
    #     "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav pitch change/**/*.wav", recursive=True)
    # all_filenames_combine_timestretch = glob.glob(
    #     "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav time stretch/**/*.wav", recursive=True)
    # all_filenames_combine_reverb = glob.glob(
    #     "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav reverb/**/*.wav", recursive=True)
    #
    # p = multiprocessing.Pool(7)
    # q = multiprocessing.Pool()
    # r = multiprocessing.Pool()
    # s = multiprocessing.Pool()
    # for file in all_filenames:
    #     p.apply_async(room_simulation, [file])
    #     p.apply_async(combine_multiple_audio, [file])
    #     p.apply_async(change_pitch_parselmouth, [file])
    #     p.apply_async(time_stretch, [file])
    #     p.apply_async(combine_hospital_audio, [file])
    #
    # for file1 in all_filenames_combine_pitch:
    #     q.apply_async(combine_pitch, [file1])
    #
    # for file2 in all_filenames_combine_timestretch:
    #     q.apply_async(combine_time_stretch, [file2])
    #
    # for file3 in all_filenames_combine_reverb:
    #     q.apply_async(combine_reverb, [file3])
    #
    # p.close()
    # # q.close()
    # # r.close()
    # # s.close()
    # p.join()
    # q.close()
    # q.join()
    # r.join()
    # s.join()
    # with multiprocessing.get_context('spawn').Pool() as pool:
    #     for file in all_filenames:
    #     # reverb_audio = pool.map_async(room_simulation, all_filenames)
    #     # combine_audio = pool.map_async(combine_multiple_audio, all_filenames)
    #     # change_pitch = pool.map_async(change_pitch_parselmouth, all_filenames)
    #     # time_stretch_wav = pool.map_async(time_stretch, all_filenames)
    #     # combine_all_pitch = pool.map_async(combine_pitch, all_filenames)
    #     # combine_all_time_stretch = pool.map_async(combine_time_stretch, all_filenames)
    #     # combine_all_reverb = pool.map_async(combine_reverb, all_filenames)
    #         reverb_audio = pool.map(room_simulation, file)
    #         combine_audio = pool.map(combine_multiple_audio, file)
    #         change_pitch = pool.map(change_pitch_parselmouth, file)
    #         time_stretch_wav = pool.map(time_stretch, file)
    #         hospital_noise = pool.map(combine_hospital_audio, file)
    #         # combine_all_pitch = pool.map(combine_pitch, file)
    #         # combine_all_time_stretch = pool.map(combine_time_stretch, file)
    #         # combine_all_reverb = pool.map(combine_reverb, file)

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Sound augmentation')
    parser.add_argument("--input_data_dir",
                        help="Path to the data directory containing files such as corpus, spk2gender, spk2utt,"
                             " utt2spk text, and wav mapping",
                        type=str,
                        default=DEFAULT_INPUT_DATA_DIR)
    parser.add_argument("--output_data_dir",
                        help="Path to the data directory containing files such as corpus, spk2gender, spk2utt,"
                             " utt2spk text, and wav mapping describing the result of augmentation",
                        type=str,
                        default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output_sound_dir",
                        help="Path to the directory that will contain the result of the augmentation",
                        type=str,
                        default=DEFAULT_OUTPUT_WAV_DIR)
    args = parser.parse_args()


    # Open utt2spk
    utt2spk_dict = open_utt2spk(ORIGINAL_UTT2SPK_FILE)
    # print(utt2spk_dict)

    # Open text
    daftar_teks_dict = open_text(ORIGINAL_TEXT_FILE)
    # print(daftar_teks_dict)

    # Open wav.scp
    wavscp_dict = open_wavscp(ORIGINAL_WAV_MAPPING_PATH)
    # print(wavscp_dict)

    # Create AudioItem list for each original audio
    original_audio_items = create_audio_item_list(ORIGINAL_WAV_MAPPING_PATH, wavscp_dict, utt2spk_dict, daftar_teks_dict)


    # Create directory for augmentation result
    augmentation_result_dir = RESULT_PATH
    if not os.path.exists(augmentation_result_dir):
        os.mkdir(augmentation_result_dir)


    # augment reverb
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav", recursive = True)
    for file in all_filenames:
        reverb_audio = room_simulation(file)

    # combine audio
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav", recursive = True)
    for file in all_filenames:
        combine_audio = combine_multiple_audio(file)

    # combine hospital noise
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav", recursive = True)
    for file in all_filenames:
        combine_hospital_noise = combine_hospital_audio(file)

    # pitch change
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav", recursive = True)
    for file in all_filenames:
        change_pitch = change_pitch_parselmouth(file)

    # time stretch
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/input data/wav/**/*.wav", recursive = True)
    for file in all_filenames:
        time_stretch_wav = time_stretch(file)

    # combine (pitch + combine audio)
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav pitch change/**/*.wav", recursive = True)
    for file in all_filenames:
        combine_all_pitch = combine_pitch(file)

    # combine (time stretch + combine audio)
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav time stretch/**/*.wav", recursive=True)
    for file in all_filenames:
        combine_all_time_stretch = combine_time_stretch(file)

    # combine (reverb + combine audio)
    all_filenames = glob.glob("/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented wav/wav reverb/**/*.wav", recursive=True)
    for file in all_filenames:
        combine_all_reverb = combine_reverb(file)

    # Augment each audio file
    generated_audio_items = list()
    for original_audio_item in original_audio_items:
        combine_generated_audio_item = combine_augment_sound(original_audio_item)
        pitch_generated_audio_item = pitch_augment_sound(original_audio_item)
        time_stretch_generated_audio_item = time_stretch_augment_sound(original_audio_item)
        reverb_generated_audio_item = reverb_augment_sound(original_audio_item)
        combine_pitch_generated_audio_item = combine_pitch_augment_sound(original_audio_item)
        combine_time_stretch_generated_audio_item = combine_time_stretch_augment_sound(original_audio_item)
        combine_reverb_generated_audio_item = combine_reverb_augment_sound(original_audio_item)

        generated_audio_items.append(combine_generated_audio_item)
        generated_audio_items.append(pitch_generated_audio_item)
        generated_audio_items.append(time_stretch_generated_audio_item)
        generated_audio_items.append(reverb_generated_audio_item)
        generated_audio_items.append(combine_pitch_generated_audio_item)
        generated_audio_items.append(combine_time_stretch_generated_audio_item)
        generated_audio_items.append(combine_reverb_generated_audio_item)

    #     # generated_audio_items.extend([combine_generated_audio_item, pitch_generated_audio_item, time_stretch_generated_audio_item, combine_pitch_generated_audio_item, combine_time_stretch_generated_audio_item]) # antara menggunakan append atau extend

    # Generate utt2spk for generated audio
    output_utt2spk_path = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented txt/utt2spk_generated.txt"
    with open(output_utt2spk_path, 'w') as generated_utt2spk_file:
        for generated_audio_item in generated_audio_items:
            generated_utt2spk_file.write(str(generated_audio_item.audio_id) + " " +str(generated_audio_item.speaker) + "\n")

    # Generate utt2spk for generated audio
    output_spk2utt_path = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented txt/spk2utt_generated.txt"
    with open(output_spk2utt_path, 'w') as generated_spk2utt_file:
        for generated_audio_item in generated_audio_items:
            generated_spk2utt_file.write(str(generated_audio_item.speaker) + " " + str(generated_audio_item.audio_id) + "\n")

    # Generate wav.csp for generated audio
    output_wavscp_path = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented txt/wav.scp_generated.txt"
    with open(output_wavscp_path, 'w') as generated_wavscp_file:
        for generated_audio_item in generated_audio_items:
            generated_wavscp_file.write(
                str(generated_audio_item.audio_id) + " " + str(generated_audio_item.audio_path))

    # Generate text for generated audio
    output_text_path = "/Users/michaeltjitra/Documents/michelle/Kerja Praktek/augmented txt/text_generated.txt"
    with open(output_text_path, 'w') as generated_text_file:
        for generated_audio_item in generated_audio_items:
            generated_text_file.write(
                str(generated_audio_item.audio_id) + " " + str(generated_audio_item.text))

    print("Processing time of %s is %.2f seconds."
          % ("audio augmentation ", time.time() - start_time))