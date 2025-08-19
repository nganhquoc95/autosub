"""
Defines autosub's main functionality.
"""

#!/usr/bin/env python

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import audioop
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import wave
import json
import requests
import time
import hashlib
import shutil
import pickle
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

from googleapiclient.discovery import build
from progressbar import ProgressBar, Percentage, Bar, ETA

from autosub.constants import (
    LANGUAGE_CODES, GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL,
)
from autosub.formatters import FORMATTERS

DEFAULT_SUBTITLE_FORMAT = 'srt'
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'en'
CACHE_ROOT = os.path.expanduser("~/.autosub_cache")


def percentile(arr, percent):
    """
    Calculate the given percentile of arr.
    """
    arr = sorted(arr)
    index = (len(arr) - 1) * percent
    floor = math.floor(index)
    ceil = math.ceil(index)
    if floor == ceil:
        return arr[int(index)]
    low_value = arr[int(floor)] * (ceil - index)
    high_value = arr[int(ceil)] * (index - floor)
    return low_value + high_value


class FLACConverter(object): # pylint: disable=too-few-public-methods
    """
    Class for converting a region of an input audio or video file into a FLAC audio file
    """
    def __init__(self, source_path, include_before=0.25, include_after=0.25, cache_dir=None):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after
        self.cache_dir = cache_dir

    def __call__(self, region, current_index=1):
        try:
            start, end = region
            start = max(0, start - self.include_before)
            end += self.include_after
            if self.cache_dir:
                temp_path = os.path.join(self.cache_dir, "region_%d.flac" % current_index)
                temp = None
            else:
                temp = tempfile.NamedTemporaryFile(suffix='.flac', delete=False)
                temp_path = temp.name
            command = ["ffmpeg", "-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", temp_path]

            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            with open(temp_path, "rb") as f:
                read_data = f.read()
            if temp:
                temp.close()
                os.unlink(temp.name)

            return read_data

        except KeyboardInterrupt:
            return None


class SpeechRecognizer(object): # pylint: disable=too-few-public-methods
    """
    Class for performing speech-to-text for an input FLAC file.
    """
    def __init__(self, language="en", rate=44100, retries=3, api_key=GOOGLE_SPEECH_API_KEY):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries

    def __call__(self, data):
        try:
            for attempt in range(self.retries):
                url = GOOGLE_SPEECH_API_URL.format(lang=self.language, key=self.api_key)
                headers = {"Content-Type": "audio/x-flac; rate=%d" % self.rate}

                try:
                    resp = requests.post(url, data=data, headers=headers)
                except requests.exceptions.ConnectionError:
                    time.sleep(10 * attempt)
                    continue

                for line in resp.content.decode('utf-8').split("\n"):
                    try:
                        line = json.loads(line)
                        line = line['result'][0]['alternative'][0]['transcript']
                        return line[:1].upper() + line[1:]
                    except IndexError:
                        # no result
                        continue
                    except JSONDecodeError:
                        continue

        except KeyboardInterrupt:
            return None


class Translator(object): # pylint: disable=too-few-public-methods
    """
    Class for translating a sentence from a one language to another.
    """
    def __init__(self, language, api_key, src, dst):
        self.language = language
        self.api_key = api_key
        self.service = build('translate', 'v2',
                             developerKey=self.api_key)
        self.src = src
        self.dst = dst

    def __call__(self, sentence):
        try:
            if not sentence:
                return None

            result = self.service.translations().list( # pylint: disable=no-member
                source=self.src,
                target=self.dst,
                q=[sentence]
            ).execute()

            if 'translations' in result and result['translations'] and \
                'translatedText' in result['translations'][0]:
                return result['translations'][0]['translatedText']

            return None

        except KeyboardInterrupt:
            return None


def which(program):
    """
    Return the path for a given executable.
    """
    def is_exe(file_path):
        """
        Checks whether a file is executable.
        """
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def extract_audio(filename, channels=1, rate=16000):
    """
    Extract audio from an input file to a temporary WAV file.
    """
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    if not os.path.isfile(filename):
        print("The given file does not exist: {}".format(filename))
        raise Exception("Invalid filepath: {}".format(filename))
    if not (which("ffmpeg") or which("ffmpeg.exe")):
        print("ffmpeg: Executable not found on machine.")
        raise Exception("Dependency not found: ffmpeg")
    command = ["ffmpeg", "-y", "-i", filename,
               "-ac", str(channels), "-ar", str(rate),
               "-loglevel", "error", temp.name]
    use_shell = True if os.name == "nt" else False
    subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
    return temp.name, rate


def find_speech_regions(filename, frame_width=4096, min_region_size=0.5, max_region_size=6): # pylint: disable=too-many-locals
    """
    Perform voice activity detection on a given audio file.
    """
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
    rate = reader.getframerate()
    n_channels = reader.getnchannels()
    chunk_duration = float(frame_width) / rate

    n_chunks = int(math.ceil(reader.getnframes()*1.0 / frame_width))
    energies = []

    for _ in range(n_chunks):
        chunk = reader.readframes(frame_width)
        energies.append(audioop.rms(chunk, sample_width * n_channels))

    threshold = percentile(energies, 0.3)

    elapsed_time = 0

    regions = []
    region_start = None

    for energy in energies:
        is_silence = energy <= threshold
        max_exceeded = region_start and elapsed_time - region_start >= max_region_size

        if (max_exceeded or is_silence) and region_start:
            if elapsed_time - region_start >= min_region_size:
                regions.append((region_start, elapsed_time))
                region_start = None

        elif (not region_start) and (not is_silence):
            region_start = elapsed_time
        elapsed_time += chunk_duration
    return regions

def generate_cache_key(source_path, src_lang, dst_lang, rate):
    h = hashlib.sha1()
    info = f"{os.path.abspath(source_path)}|{src_lang}|{dst_lang}|{rate}"
    h.update(info.encode("utf-8"))
    return h.hexdigest()

def load_pickle(path):
    """Load Python object has been pickle into file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def dump_pickle(path, obj):
    """Write Python object into file by pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_subtitles( # pylint: disable=too-many-locals,too-many-arguments
        source_path,
        output=None,
        concurrency=DEFAULT_CONCURRENCY,
        src_language=DEFAULT_SRC_LANGUAGE,
        dst_language=DEFAULT_DST_LANGUAGE,
        subtitle_file_format=DEFAULT_SUBTITLE_FORMAT,
        api_key=None,
        no_cache=False,
    ):
    """
    Given an input audio/video file, generate subtitles in the specified language and format.
    """
    audio_filename, audio_rate = extract_audio(source_path)

    cache_key = generate_cache_key(source_path, src_language, dst_language, audio_rate)
    cache_dir = os.path.join(CACHE_ROOT, cache_key)
    state_file = os.path.join(cache_dir, "state.pickle")
    current_index = 1

    # Respect no_cache flag: do not create or use cache when requested
    if no_cache:
        cache_dir = None
        state_file = None
    else:
        # Ensure cache dir exists
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
            except OSError:
                # ignore races or permission errors; proceed without caching if cannot create
                cache_dir = None
                state_file = None

    regions = find_speech_regions(audio_filename)

    pool = multiprocessing.Pool(concurrency)
    converter = FLACConverter(source_path=audio_filename, cache_dir=cache_dir)
    recognizer = SpeechRecognizer(language=src_language, rate=audio_rate)

    transcripts = []
    # Load cached state if available and matches current regions
    state = None
    if state_file and os.path.exists(state_file):
        try:
            state = load_pickle(state_file)
            if state.get('regions') == regions:
                transcripts = state.get('transcripts', [])
            else:
                state = None
        except Exception:
            state = None

    pbar = None
    try:
        if regions:
            widgets = ["Converting speech regions to FLAC files: ", Percentage(), ' ', Bar(), ' ',
                       ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_regions = []
            
            # Load from cache
            if cache_dir:
                flac_files = [f for f in os.listdir(cache_dir) if f.startswith("region_") and f.endswith(".flac")]
                if len(flac_files) == len(regions):  
                    for i in range(len(regions)):
                        flac_path = os.path.join(cache_dir, ("region_%d.flac" % i))
                        if os.path.exists(flac_path):
                            with open(flac_path, 'rb') as fh:
                                extracted_regions.append(fh.read())
            # convert to flac files if there is no cache
            if len(extracted_regions) == 0:
                # for i, extracted_region in enumerate(pool.imap(converter, regions)):
                #     extracted_regions.append(extracted_region)
                #     pbar.update(i)
                for i in range(len(regions)):
                    extracted_region = converter(regions[i], i)
                    extracted_regions.append(extracted_region)
                    pbar.update(i)
            pbar.finish()

            dump_pickle(state_file, {
                'regions': regions,
                'transcripts': transcripts,
                'translated': state.get('translated') if state else None
            })

            widgets = ["Performing speech recognition: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()

            for i in range(len(transcripts) ,len(extracted_regions)):
                transcript = recognizer(extracted_regions[i])
                transcripts.append(transcript)
                pbar.update(i)

                if i % 10 == 0:
                    if state_file:
                        try:
                            dump_pickle(state_file, {
                                'regions': regions,
                                'transcripts': transcripts,
                                'translated': state.get('translated') if state else None
                            })
                        except Exception:
                            pass
            pbar.finish()

            if src_language.split("-")[0] != dst_language.split("-")[0]:
                if api_key:
                    google_translate_api_key = api_key
                    translator = Translator(dst_language, google_translate_api_key,
                                            dst=dst_language,
                                            src=src_language)
                    prompt = "Translating from {0} to {1}: ".format(src_language, dst_language)
                    widgets = [prompt, Percentage(), ' ', Bar(), ' ', ETA()]
                    pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
                    translated_transcripts = []
                    if state_file and state and state.get('translated') and len(state.get('translated')) == len(regions):
                        translated_transcripts = state.get('translated')

                    for i in range(len(translated_transcripts), len(transcripts)):
                        transcript = translator(transcripts[i])
                        translated_transcripts.append(transcript)
                        pbar.update(i)
                        
                        if state_file:
                            try:
                                dump_pickle(state_file, {
                                    'regions': regions,
                                    'transcripts': transcripts,
                                    'translated': translated_transcripts
                                })
                            except Exception:
                                pass
                    pbar.finish()
                    transcripts = translated_transcripts
                else:
                    print(
                        "Error: Subtitle translation requires specified Google Translate API key. "
                        "See --help for further information."
                    )
                    return 1

    except KeyboardInterrupt:
        if pbar:
            pbar.finish()
        pool.terminate()
        pool.join()
        print("Cancelling transcription")
        # persist current state before exiting so resume is possible
        if state_file:
            try:
                dump_pickle(state_file, {
                    'regions': regions,
                    'transcripts': transcripts,
                    'translated': state.get('translated') if state else None
                })
            except Exception:
                pass
        raise

    timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
    formatter = FORMATTERS.get(subtitle_file_format)
    if not formatter:
        raise Exception(f"Unknown subtitle format: {subtitle_file_format}")
    formatted_subtitles = formatter(timed_subtitles)

    dest = output

    if not dest:
        base = os.path.splitext(source_path)[0]
        dest = "{base}.{format}".format(base=base, format=subtitle_file_format)

    with open(dest, 'wb') as output_file:
        output_file.write(formatted_subtitles.encode("utf-8"))

    try:
        os.remove(audio_filename)
        if cache_dir and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception:
        # If cleanup fails, ignore and continue — cache is non-critical
        pass

    return dest


def validate(args):
    """
    Check that the CLI arguments passed to autosub are valid.
    """
    if args.format not in FORMATTERS:
        print(
            "Subtitle format not supported. "
            "Run with --list-formats to see all supported formats."
        )
        return False

    if args.src_language not in LANGUAGE_CODES.keys():
        print(
            "Source language not supported. "
            "Run with --list-languages to see all supported languages."
        )
        return False

    if args.dst_language not in LANGUAGE_CODES.keys():
        print(
            "Destination language not supported. "
            "Run with --list-languages to see all supported languages."
        )
        return False

    if not args.source_path:
        print("Error: You need to specify a source path.")
        return False

    return True


def main():
    """
    Run autosub as a command-line program.
    """
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', help="Path to the video or audio file to subtitle",
                        nargs='?')
    parser.add_argument('-C', '--concurrency', help="Number of concurrent API requests to make",
                        type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument('-o', '--output',
                        help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)")
    parser.add_argument('-F', '--format', help="Destination subtitle format",
                        default=DEFAULT_SUBTITLE_FORMAT)
    parser.add_argument('-S', '--src-language', help="Language spoken in source file",
                        default=DEFAULT_SRC_LANGUAGE)
    parser.add_argument('-D', '--dst-language', help="Desired language for the subtitles",
                        default=DEFAULT_DST_LANGUAGE)
    parser.add_argument('-K', '--api-key',
                        help="The Google Translate API key to be used. \
                        (Required for subtitle translation)")
    parser.add_argument('--list-formats', help="List all available subtitle formats",
                        action='store_true')
    parser.add_argument('--list-languages', help="List all available source/destination languages",
                        action='store_true')
    parser.add_argument('--no-cache', help="Disable cache (do not read/write cache files)",
                        action='store_true')

    args = parser.parse_args()

    if args.list_formats:
        print("List of formats:")
        for subtitle_format in FORMATTERS:
            print("{format}".format(format=subtitle_format))
        return 0

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(LANGUAGE_CODES.items()):
            print("{code}\t{language}".format(code=code, language=language))
        return 0

    if not validate(args):
        return 1

    try:
        subtitle_file_path = generate_subtitles(
            source_path=args.source_path,
            concurrency=args.concurrency,
            src_language=args.src_language,
            dst_language=args.dst_language,
            api_key=args.api_key,
            subtitle_file_format=args.format,
            output=args.output,
            no_cache=args.no_cache,
        )
        print("Subtitles file created at {}".format(subtitle_file_path))
    except KeyboardInterrupt:
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
