import os
from datetime import timedelta
import re
from typing import Optional, Generator

import srt
import lxml
import wave
import chardet

from bs4 import BeautifulSoup
from pydub import AudioSegment

ARCHIMOB_XML_DIR = 'Archimob/Archimob_Release_2'
ARCHIMOB_WAV_DIR = 'Archimob/archimob_r2_audio_share/audio_segmented_anonymized'
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xml': 'http://www.w3.org/XML/1998/namespace'
}

# ARCHIMOB FUNCTIONS


def get_archimob_data(recording: str, start: str, end: str, output_dir=None) -> tuple[str, str]:
    """Extract audio and text  snippets from the ArchiMob dataset."""
    audio_files = []
    audio_directory = os.path.join(ARCHIMOB_WAV_DIR, recording)
    all_files = os.listdir(audio_directory)
    for i in range(int(start), int(end) + 1):
        audio_file = [file for file in all_files if file.endswith(f'_{i}.wav')]
        if len(audio_file) == 1:
            audio_files.append(os.path.join(audio_directory, audio_file[0]))
        elif len(audio_file) == 0:
            print(f'No audio file found for {recording}_{i}')
        else:
            print(f'Multiple audio files found for {recording}_{i}')
    output_file = f'{recording}_{start}-{end}.wav'
    audio_file, tokens = get_archimob_transcription(
        audio_files, os.path.join(output_dir, output_file))
    tokens_file = write_tokens_to_file(tokens, os.path.join(
        output_dir, f'{recording}_{start}-{end}.txt'))
    return audio_file, tokens_file


def get_archimob_transcription(audio_files: list[str], output_file: str) -> tuple[str, list[str]]:
    """Extract text snippets from the ArchiMob dataset."""
    all_tokens = []
    for audio_file in audio_files:
        file_name = audio_file.rstrip('.wav')
        dir_name = os.path.dirname(file_name)
        folder = os.path.basename(dir_name)
        sent_id = os.path.basename(file_name)
        xml_file = os.path.join(ARCHIMOB_XML_DIR, f'{folder}.xml')
        # 'd1082_1_TLI_21' --> 'd1082_1-TLI_21'
        sent_id = sent_id.replace('_', '-', 2)
        sent_id = sent_id.replace('-', '_', 1)
        tokens = extract_tokens_from_tei(xml_file, sent_id)
        all_tokens.extend(tokens)

    if len(audio_files) > 1:
        concated_audio = concat_audio_files(audio_files, output_file)
    else:
        concated_audio = audio_files[0]
    return concated_audio, all_tokens


# def extract_tokens_from_tei(xml_file: str, sent_id=None) -> list[str]:
#     with open(xml_file, 'r', encoding='utf-8') as f:
#         soup = BeautifulSoup(f, 'html.parser')
#         # get all text from xml file
#         if sent_id is None:
#             soup.get_text()
#             tokens = text.split('\n')
#         else:
#             # get u element
#             u = soup.find('u', start=f'media_pointers#{sent_id}')
#             text = u.get_text()
#             tokens = text.split('\n')
#             # remove empty strings
#             tokens = [token for token in tokens if token]

#         return tokens


# SWISSTEXT AD FUNCTIONS
def extract_audio_and_text_span(wav_file: str, srt_file: str, start_idx: int, end_idx: int, outdir: str) -> tuple[str, str]:
    """Extract a snippet of audio from a WAV file and corresponding text from a SRT file."""
    audio_files = []
    text = []
    filename = os.path.basename(wav_file).rstrip('.wav')

    # Set up temp dir for audio snippets to be stored before concatenation
    tmp_dir = os.path.join(outdir, f'{filename}_tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Parse SRT file and extract audio snippets
    srt_data = read_srt_file(srt_file, start_idx, end_idx)
    timestamps = [(sub['start'], sub['end']) for sub in srt_data]
    text = [sub['text'] for sub in srt_data]
    audio_files = extract_audio_snippets(wav_file, timestamps, tmp_dir)

    # Rename and if necessary concatenate audio snippets
    output_audio = f'{start_idx}-{end_idx}_{filename}.wav'
    output_audio_path = os.path.join(outdir, output_audio)

    if len(audio_files) > 1:
        concat_audio_files(audio_files, output_audio_path)
    else:
        raise ValueError('Error extracting audio snippets')

    # Delete temporary directory and files
    for file in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir)

    # Write text to file, one sentence per line
    output_text_file = output_audio_path.replace('.wav', '.txt')
    write_tokens_to_file(text, output_text_file)
    return output_audio_path, output_text_file


def extract_audio_and_text_all(wav_file: str, srt_file: str, outdir: str, prefix: Optional[str]) -> list[tuple[str, str]]:
    """Extract all audio snippets appearing in a SRT file and writes both audio and text to files."""
    audio_files = []
    basename = os.path.basename(wav_file).rstrip('.wav')
    print(f'Processing {basename}...')

    if already_extracted(srt_file, basename, outdir):
        print(f'Files for {basename} already extracted.')
        return []
    else:
        # remove all partial files
        for file in os.listdir(outdir):
            if basename in file:
                os.remove(os.path.join(outdir, file))
                print(f'Removed {file}')

    srt_data = read_srt_file(srt_file)
    timestamps = [(sub['start'], sub['end'], sub['idx']) for sub in srt_data]
    audio_files = extract_audio_snippets(wav_file, timestamps, outdir)

    for idx, audio_file in enumerate(audio_files):
        new_name = f'{prefix}_{idx+1}_{basename}.wav'
        os.rename(audio_file, os.path.join(outdir, new_name))

    text_files = []
    for d in srt_data:
        output_text_file = os.path.join(outdir, f'{prefix}_{d["idx"]}_{
                                        basename}.txt') if prefix else os.path.join(outdir, f'{d["idx"]}_{basename}.txt')
        text_outfile = write_tokens_to_file([d['text']], output_text_file)
        text_files.append(text_outfile)

    return list(zip(audio_files, text_files))


def already_extracted(srt_file: str, basename: str, outdir: str) -> bool:
    """Check if audio and text files have already been extracted for a given basename."""
    audio_files = []
    text_files = []
    for file in os.listdir(outdir):
        if basename in file:
            if file.endswith('.wav'):
                audio_files.append(file)
            elif file.endswith('.txt'):
                text_files.append(file)
    processed = len(audio_files) > 0 and len(text_files) > 0
    processed = processed and len(audio_files) == len(text_files)
    nr = get_end_idx(srt_file)
    processed = processed and len(audio_files) == nr
    return processed


def read_srt_file(srt_file: str, start_idx: int = 0, end_idx: int = -1) -> list[dict[str, str]]:
    """Read an SRT file and return a list of dictionaries with subtitle data."""
    encoding = detect_encoding(srt_file)
    with open(srt_file, 'r', encoding=encoding) as f:
        srt_content = f.read()

        if detect_swiss_srt_format(srt_content):
            srt_generator = parse_swiss_srt(srt_content)
        else:
            srt_generator = srt.parse(srt_content)

        srt_data = []
        offset = timedelta(seconds=0)

        # TODO: Find a better solution
        if end_idx == -1:
            end_idx = 999999
        prev_idx = 0
        index_anomaly = False
        for sub in srt_generator:
            # Check if the first subtitle starts at 10 hours --> weird time offset in srf data
            if sub.index == 1:
                if sub.start.seconds >= 36000:
                    offset = timedelta(seconds=36000)

            if prev_idx > sub.index and not index_anomaly:
                index_anomaly = True
                print(f'Anomaly in SRT file: {
                      srt_file}, continuing indexing...')

            if sub.index < start_idx:
                continue
            elif sub.index <= end_idx:
                start = srt.timedelta_to_srt_timestamp(sub.start - offset)
                end = srt.timedelta_to_srt_timestamp(sub.end - offset)
                text = sub.content.replace('\n', ' ')
                srt_data.append(
                    {'start': start,
                     'end': end,
                     'text': text,
                     'idx': sub.index if not index_anomaly else prev_idx + 1})

            if index_anomaly:
                prev_idx += 1
            else:
                prev_idx = sub.index
    return srt_data


def process_srfad_files(wav_dir: str, srt_dir: str, manual_match: dict) -> list[tuple[str, str]]:
    """ Find and match SRFAD audio and SRT files, to be further processed."""
    data = []
    idx = 0
    for file in os.listdir(wav_dir):
        if file.endswith('.wav'):
            wav_file = os.path.join(wav_dir, file)

            srt_file = os.path.join(srt_dir, file.replace('.wav', '.srt'))
            srt_file_2 = os.path.join(srt_dir, file.replace('.wav', '.txt'))
            srt_file_3 = os.path.join(
                srt_dir, file.replace('.wav', '.CHDE.srt'))
            srt_file_matched = None
            if os.path.exists(srt_file):
                srt_file_matched = srt_file
            elif os.path.exists(srt_file_2):
                srt_file_matched = srt_file_2
            elif os.path.exists(srt_file_3):
                srt_file_matched = srt_file_3
            else:
                if file in manual_match:
                    srt_file_matched = os.path.join(
                        srt_dir, manual_match[file])
                else:
                    print(f'No SRT file found for {wav_file}')
            if srt_file_matched:
                data.append((wav_file, srt_file_matched))
                idx += 1
    return data


def get_end_idx(srt_file: str) -> int:
    """Find the index of the last subtitle in an SRT file, i.e. find the number of subtitles."""
    # encoding = detect_encoding(srt_file) takes way too long
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except UnicodeDecodeError:
        with open(srt_file, 'r', encoding='iso-8859-1') as f:
            srt_content = f.read()

    if detect_swiss_srt_format(srt_content):
        srt_generator = parse_swiss_srt(srt_content)
    else:
        srt_generator = srt.parse(srt_content)

    sub = None
    for sub in srt_generator:
        pass

    if sub is not None:
        return sub.index
    else:
        return -1


def prepare_srfad_data(wav_dir: str, srt_dir: str, outdir: str, manual_match: dict) -> None:
    """Prepare SRFAD data for further processing by extracting audio and text snippets."""
    data_files = process_srfad_files(wav_dir, srt_dir, manual_match)
    idx = 0
    for wav_file, srt_file in data_files:
        idx += 1
        id_string = str(idx).zfill(3)
        extract_audio_and_text_all(
            wav_file, srt_file, outdir, prefix=id_string)


class Subtitle:
    def __init__(self, index, start, end, content):
        self.index = index
        self.start = start
        self.end = end
        self.content = content


def parse_swiss_srt(srt_content) -> Generator[Subtitle, None, None]:
    """
    Parses Swiss-style subtitles into a generator of Subtitle objects.
    Swiss-style format has timestamps like HH:MM:SS:FF and spaces in between.
    """
    # Regex pattern to match Swiss-style timestamp lines (start and end times)
    timestamp_pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}:\d{2})\s*(\d{2}:\d{2}:\d{2}:\d{2})')
    # Regex pattern to match subtitle index
    index_pattern = re.compile(r'^\d+$')

    lines = srt_content.splitlines()
    index = None
    start = None
    end = None
    content = ''

    for line in lines:
        if index_pattern.match(line):
            index = int(index_pattern.match(line).group())
        elif timestamp_pattern.match(line):
            timestamp_match = timestamp_pattern.match(line)
            start_time_str = timestamp_match.group(1)
            end_time_str = timestamp_match.group(2)
            start = parse_swiss_time_to_timedelta(start_time_str)
            end = parse_swiss_time_to_timedelta(end_time_str)
        else:
            content += line

        if index and start and end and content:
            yield Subtitle(index, start, end, content)
            index = None
            start = None
            end = None
            content = ''


def detect_swiss_srt_format(srt_content):
    """
    Detects whether the SRT file is in Swiss-style format.
    """
    srt_lines = srt_content.splitlines()
    # Check if the SRT file is in Swiss-style format
    second_line = srt_lines[1]
    if re.search(r'\d{2}:\d{2}:\d{2}:\d{2}', second_line):
        return True
    return False


def parse_swiss_time_to_timedelta(time_str: str):
    """
    Converts Swiss-style timestamp HH:MM:SS:FF to a timedelta object.
    """
    hours, minutes, seconds, frames = map(int, time_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds + \
        frames / 24  # Assuming 24 fps for the frames
    return timedelta(seconds=total_seconds)


# AUDIO FUNCTIONS
def concat_audio_files(audio_files: list[str], output_file: str) -> str:
    """Concatenate multiple audio files into a single file."""
    data = []
    for clip in audio_files:
        w = wave.open(clip, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    output = wave.open(output_file, "wb")
    output.setparams(data[0][0])
    for d in data:
        output.writeframes(d[1])
    output.close()
    return output_file


def extract_audio_snippet(audio_file: str, start: str, end: str, output_file: str) -> str:
    """Extract a snippet of audio from an audio file.

    Args:
        audio_file (str): Path to the input audio file (WAV or MP3 format).
        start (str): Start time in HH:MM:SS,SSS format (e.g., 10:00:46,320).
        end (str): End time in HH:MM:SS,SSS format (e.g., 10:00:53,480).
        output_file (str): Path to the output audio file (WAV format).
    """
    start_sec = time_to_seconds(start) * 1000  # convert to milliseconds
    end_sec = time_to_seconds(end) * 1000  # convert to milliseconds

    audio = AudioSegment.from_file(audio_file)
    snippet = audio[start_sec:end_sec]
    snippet.export(output_file, format="wav")

    return output_file


def extract_audio_snippets(audio_file: str, time_stamps: list[tuple[str, str, str]], output_dir: str) -> list[str]:
    """Extract audio snippets from an audio file given a list of time stamps."""
    outfiles = []
    audio = AudioSegment.from_file(audio_file)
    for start, end, idx in time_stamps:
        start_sec = time_to_seconds(start) * 1000
        end_sec = time_to_seconds(end) * 1000
        snippet = audio[start_sec:end_sec]
        audio_basename = os.path.basename(audio_file)
        outfile = os.path.join(output_dir, f'{idx}_{audio_basename}')
        snippet.export(outfile, format="wav")
        outfiles.append(outfile)
    return outfiles


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in minutes."""
    audio = AudioSegment.from_wav(file_path)
    # pydub gives length in milliseconds
    duration_in_minutes = len(audio) / (1000 * 60)
    return duration_in_minutes


def calculate_total_audio_duration(folder_path):
    """Calculate the total duration of all audio files in a folder."""
    total_duration = 0.0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                total_duration += get_audio_duration(file_path)
    return total_duration

# GENERAL HELPERS


def time_to_seconds(time_str: str) -> float:
    """Convert time in HH:MM:SS,SSS format to total seconds."""
    hours, minutes, seconds = time_str.split(':')
    seconds, millis = seconds.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * \
        60 + int(seconds) + int(millis) / 1000
    return total_seconds


def write_tokens_to_file(tokens: list[str], output_file: str) -> str:
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')

    return output_file


def detect_encoding(file_path):
    """Detects the encoding of a text file."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']


def basic_tokenizer(text: str) -> list[str]:
    """Basic tokenization of a text string, considering punctuation and whitespace."""
    tokens = text.split([' ', '.', ',', '!', '?', ';', ':',
                         '(', ')', '[', ']', '{', '}', '<', '>', '"', "'"])
    return tokens


def main():
    # output_dir = 'data_samples'
    # # audio_file, tokens_file = get_archimob_data('1082_1', 21, 40, output_dir)

    # srt_file = 'ad_swisstext_srg/sample/1G100_365.CHDE.srt'
    # wav_file = 'ad_swisstext_srg/sample/1G100_365.wav'

    # outfile, text = get_swisstext_data(wav_file, srt_file, 1, 20, output_dir)

    # 968.6078499999999
    # path = "/home/vera/Documents/Uni/Master/Master_Thesis/data/PATZeK/selected_audio"
    # print(calculate_total_audio_duration(path))

    # 13.364400000000005
    # path2 = '/home/vera/Documents/Uni/Master/Master_Thesis/data/CRM_Swiss/CRM_Swiss/data_CRM_dialect'
    # print(calculate_total_audio_duration(path2))

    # 152.2712166666669
    # path3 = '/home/vera/Documents/Uni/Master/Master_Thesis/data/BKB_Swiss/BKB_Swiss/data'
    # print(calculate_total_audio_duration(path3))

    # 220.34396666666706
    # path4 = '/home/vera/Documents/Uni/Master/Master_Thesis/data/ArticulatoryObstruction/ArticulatoryObstruction/bySentence'
    # print(calculate_total_audio_duration(path4))

    # # 25.13396666666665
    # path5 = '/home/vera/Documents/Uni/Master/Master_Thesis/data/TEVOID/TEVOID/1_spntLong_16spkrs'
    # print(calculate_total_audio_duration(path5))

    path6 = '/home/vera/Documents/Uni/Master/Master_Thesis/data/TEVOID/TEVOID'
    print(calculate_total_audio_duration(path6))


if __name__ == "__main__":

    main()
