import os
import wave
import io
import logging
from datetime import timedelta
import re
from typing import Optional
from dataclasses import dataclass

import srt
import chardet
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pydub import AudioSegment
from lxml import etree

ARCHIMOB_XML_DIR = 'Archimob/Archimob_Release_2'
ARCHIMOB_WAV_DIR = 'Archimob/archimob_r2_audio_share/audio_segmented_anonymized'
NAMESPACES = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xml': 'http://www.w3.org/XML/1998/namespace'
}

log_dir = '../log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'datareader.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logging.basicConfig(filename='datareader.log', level=logging.INFO)

# MUNDARTKORPUS FUNCTIONS
def extract_text_chmk(xml_file: str) -> list[str]:
    """Extracts sentences from a structured TEI XML file."""
    punctuation = ['.', '!', '?', ',', ';', ':']
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Define the namespace (TEI uses a default namespace)
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    sentences = []

    # Find all sentence elements <s> inside paragraphs <p>
    for s in root.xpath('.//tei:text/tei:body//tei:p/tei:s', namespaces=ns):
        sentence = ''
        words = [word.text for word in s.xpath('.//tei:w', namespaces=ns) if word.text]
        for w in words: 
            if len(w) == 1 and w in punctuation:
                sentence = sentence.strip()
            sentence += w + ' '           
        sentences.append(sentence.strip())  
    return sentences


def process_chmk(xml_dir: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            xml_file = os.path.join(xml_dir, file)
            sentences = extract_text_chmk(xml_file)
            with open(os.path.join(outdir, file.replace('.xml', '.txt')), 'w', encoding='utf-8') as f:
                for sent in sentences:
                    f.write(sent + '\n')


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
def extract_audio_and_text_all(wav_file: str, srt_file: str, outdir: str, prefix: Optional[str]) -> list[tuple[str, str]]:
    """Extract all audio snippets appearing in a SRT file and writes both audio and text to files."""
    # check if wav file ends in .wav
    if not wav_file.endswith('.wav'):
        raise ValueError(
            'Looks like the audio file is not passed correctly, please provide a .wav file.')
    audio_files = []
    basename = os.path.basename(wav_file).rstrip('.wav')
    logging.info(f'Processing {basename}...')

    if already_extracted(srt_file, basename, outdir):
        logging.info(f'Files for {basename} already extracted.')
        return []
    else:
        # remove all partial files
        for file in os.listdir(outdir):
            if basename in file:
                os.remove(os.path.join(outdir, file))
                logging.info(f'Removed {file} for reprocessing')

    srt_data = SRTReader(srt_file)

    timestamps = []
    text_files = []
    for d in srt_data:
        output_text_file = os.path.join(outdir, f'{prefix}_{d.index}_{
                                        basename}.txt') if prefix else os.path.join(outdir, f'{d.index}_{basename}.txt')
        
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(d.content)
        text_files.append(output_text_file)
        timestamps.append((d.start, d.end, d.index))

    # Extract audio snippet    
    audio_files = extract_audio_snippets(wav_file, timestamps, outdir, prefix=prefix)
    return list(zip(audio_files, text_files))


def match_audio_srt_files(wav_dir: str, srt_dir: str, manual_match: dict) -> list[tuple[str, str]]:
    """ Find and match SRFAD audio and SRT files, to be further processed."""
    data = []
    idx = 0
    for file in os.listdir(wav_dir):
        if file.endswith('.wav'):
            wav_file = os.path.join(wav_dir, file)
            if file in manual_match:
                srt_file_matched = os.path.join(
                    srt_dir, manual_match[file])
            else:
                srt_file = os.path.join(srt_dir, file.replace('.wav', '.srt'))
                srt_file_2 = os.path.join(
                    srt_dir, file.replace('.wav', '.txt'))
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
                    logging.warning(f'No SRT file found for {wav_file}')

            if srt_file_matched:
                data.append((wav_file, srt_file_matched))
                idx += 1
    return data


def prepare_srfad_data(wav_dir: str, srt_dir: str, outdir: str, manual_match: dict) -> None:
    """Prepare SRFAD data for further processing by extracting audio and text snippets."""
    data_files = match_audio_srt_files(wav_dir, srt_dir, manual_match)
    idx = 0
    with tqdm(data_files, desc="Processing files") as pbar:
        for wav_file, srt_file in pbar:
            idx += 1
            id_string = str(idx).zfill(3)
            extract_audio_and_text_all(
                wav_file, srt_file, outdir, prefix=id_string)


def already_extracted(srt_file: str, basename: str, outdir: str) -> bool:
    """Check if audio and text files have already been extracted for a given basename."""
    srt_reader = SRTReader(srt_file)
    audio_files = []
    text_files = []
    # create outdir if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for file in os.listdir(outdir):
        if basename in file:
            if file.endswith('.wav'):
                audio_files.append(file)
            elif file.endswith('.txt'):
                text_files.append(file)
    processed = len(audio_files) > 0 and len(text_files) > 0
    processed = processed and len(audio_files) == len(text_files)
    processed = processed and len(audio_files) == srt_reader.nr_valid_subtitles
    return processed

@dataclass
class Subtitle:
    index: int
    start: timedelta
    end: timedelta
    content: str


class SRTReader():

    def __init__(self, srt_file):
        self.srt_file = srt_file
        self.encoding = detect_encoding(srt_file)
        self.is_swiss_srt = self._detect_swiss_srt_format()

        self.data = []
        self.nr_valid_subtitles = 0
        self.has_offset = False
        self.first_index = -1

        self.parse_file()

    def __iter__(self):
        for sub in self.data:
            sub.content = self.clean_subtitle_text(sub.content)
            yield sub

    def __repr__(self):
        return f'SRTReader(srt_file={self.srt_file})'

    def _detect_swiss_srt_format(self):
        """
        Detects whether the SRT file is in Swiss-style format via the timestamp format
            swiss srt: 00:00:00:00
            normal srt: 00:00:00,000
        """
        with open(self.srt_file, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
            second_line = lines[1]
            # check format of timestamps:
            if re.search(r'\d{2}:\d{2}:\d{2}:\d{2}', second_line):
                return True
            return False

    def _parse_swiss_time_to_timedelta(self, time_str: str, offset: bool = False) -> timedelta:
        """
        Converts Swiss-style timestamp HH:MM:SS:FF to a timedelta object.
        """
        hours, minutes, seconds, frames = map(int, time_str.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds + \
            frames / 24  # Assuming 24 fps for the frames
        if offset:
            # remove 10 hours offset
            total_seconds -= 36000
        return timedelta(seconds=total_seconds)

    def parse_file_swiss(self, srt_file) -> list[Subtitle]:
        # Regex pattern to match Swiss-style timestamp lines (start and end times)
        timestamp_pattern = re.compile(
            r'(\d{2}:\d{2}:\d{2}:\d{2})\s*(\d{2}:\d{2}:\d{2}:\d{2})')
        # Regex pattern to match subtitle index
        index_pattern = re.compile(r'^\d+$')

        index = None
        start = None
        end = None
        content = ''
        subtitles = []
        with open(srt_file, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
            for line in lines:
                if index_pattern.match(line):
                    index = int(index_pattern.match(line).group())

                    # Check for case where indexing is started again mid-file
                    prev_idx = subtitles[-1].index if subtitles else 0
                    if index < prev_idx:
                        logging.info(
                            f'Anomaly in SRT file: {srt_file}, continuing indexing...')
                        index = prev_idx + 1

                elif timestamp_pattern.match(line):
                    timestamp_match = timestamp_pattern.match(line)
                    start_time_str = timestamp_match.group(1)
                    end_time_str = timestamp_match.group(2)

                    # Check if there's this weird 10h offset
                    if index == 0 or index == 1:
                        if start_time_str.startswith('10:'):
                            self.has_offset = True
                    start = self._parse_swiss_time_to_timedelta(
                        start_time_str, self.has_offset)
                    end = self._parse_swiss_time_to_timedelta(
                        end_time_str, self.has_offset)
                else:
                    content += '\n' + line.rstrip()

                all_data_collected = (index is not None) and (
                    start is not None) and (end is not None) and (content != '')

                if (line.rstrip() == '' and all_data_collected) or (line == lines[-1] and all_data_collected):
                    content = content.replace('\n', ' ').rstrip()
                    subtitle = Subtitle(index, start, end, content)
                    if self.subtitle_is_valid(subtitle):
                        subtitles.append(subtitle)
                    index = None
                    start = None
                    end = None
                    content = ''
        return subtitles

    def parse_file_normal(self, srt_file: str) -> list[Subtitle]:
        data = []
        with open(srt_file, 'r', encoding=self.encoding) as f:
            srt_generator = srt.parse(f.read())
            for sub in srt_generator:
                index = sub.index
                # Check if the first subtitle starts at 10 hours --> weird time offset in srf data
                if sub.index == 0 or sub.index == 1:
                    if sub.start.seconds >= 36000:
                        self.has_offset = True

                # Check for case where indexing is started again mid-file
                prev_idx = data[-1].index if data else 0
                if index < prev_idx:
                    index = prev_idx + 1
                    logging.info(
                        f'Anomaly in SRT file: {srt_file}, continuing indexing...')

                start = sub.start
                end = sub.end
                if self.has_offset:
                    offset = timedelta(seconds=36000)
                    start = sub.start - offset
                    end = sub.end - offset

                text = sub.content.replace('\n', ' ')
                subtitles = Subtitle(index, start, end, text)
                if self.subtitle_is_valid(subtitles):
                    data.append(subtitles)

        return data

    def parse_file(self):
        if self.is_swiss_srt:
            self.data = self.parse_file_swiss(self.srt_file)
        else:
            self.data = self.parse_file_normal(self.srt_file)

        self.nr_valid_subtitles = len(self.data)
        self.first_index = self.data[0].index

    def subtitle_is_valid(self, sub: Subtitle) -> bool:
        """Check if a subtitle is valid."""
        # Faulty case 1: empty content
        if sub.content.isspace() or sub.content == '':
            return False
        # Faulty case 2: faulty timestamps (start > end or start == end)
        elif sub.start >= sub.end:
            return False
        else:
            return True

    def clean_subtitle_text(self, text: str) -> str:
        """Clean subtitle text by removing unwanted characters."""

        orig = text
        changed = False
        # there is brackets with timestamps (some text 00:00:02 kj)
        text = re.sub(r'\(.*?\d{2}:\d{2}:\d{2}.*?\)', '', text)
        # reaplace (über .+?) with ''
        text = re.sub(r'\(über .+?\)', '', text, flags=re.IGNORECASE)

        # 1. Trailing $
        if text.startswith('$$'):
            text = text[2:]
        if text.startswith('$'):
            text = text[1:]
        if text.startswith('*'):
            text = text[1:]

        # there is double brackets
        text = re.sub(r'\(\(.*?\)\)', '', text)

        # remove brackets but keep content
        # get content between brackets
        matches = re.findall(r'\((.*?)\)', text)
        for content in matches:
            # if content is all uppercase, remove brackets and content
            if content.isupper():
                text = re.sub(r'\(' + content + r'\)', '', text)
            # if content is within " ", remove brackets and content
            elif content.startswith('"') and content.endswith('"'):
                text = re.sub(r'\(' + content + r'\)', '', text)
        

        text = re.sub(r'\(', '', text)
        text = re.sub(r'\)', '', text)

        # Remove * and $ when whitespace before and after
        text = re.sub(r'\s\*\s', ' ', text)
        text = re.sub(r'\s\$\s', ' ', text)


        # remove ...
        text = re.sub(r'\.\.\.', '', text)

        # Trailing $
        if text.startswith('$$'):
            text = text[2:]
        if text.startswith('$'):
            text = text[1:]
        if text.startswith('*'):
            text = text[1:]

        # check if changed before removing whitespaces --> no logging if only whitespaces are removed
        if orig != text:
            changed = True

        # remove several whitespaces and leading/trailing whitespaces
        text = re.sub(r'\s+', ' ', text)
        final = text.strip().lstrip(' ')

        if changed:
            logging.info(f'Changed text: {orig} --> {text}')
       
        return final


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


def extract_audio_snippets(audio_file: str, time_stamps: list[tuple[timedelta, timedelta, int]], output_dir: str, prefix: Optional[str] = None) -> list[str]:
    """Extract audio snippets from an audio file given a list of timestamps using timedelta. (Loads the audio file only once)"""
    outfiles = []
    audio = AudioSegment.from_file(audio_file)

    for start, end, idx in time_stamps:
        start_ms = int(start.total_seconds() * 1000)
        end_ms = int(end.total_seconds() * 1000)

        if start_ms > len(audio):
            raise ValueError(f'Timestamp out of bounds: {audio_file}')
        
        # check if it's longer than 30 seconds
        if end_ms - start_ms > 30000:
            logging.warning(f'Audio snippet too long: {audio_file}, skipping...')
            continue

        snippet = audio[start_ms:end_ms]
        audio_basename = os.path.basename(audio_file)
        filename = f"{prefix}_{idx}_{audio_basename}" if prefix else f"{idx}_{audio_basename}"
        outfile = os.path.join(output_dir, filename)

        snippet.export(outfile, format="wav")
        outfiles.append(outfile)

    return outfiles


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in minutes."""
    audio = AudioSegment.from_wav(file_path)
    # pydub gives length in milliseconds
    duration_in_minutes = len(audio) / (1000 * 60)
    return duration_in_minutes


def calculate_total_audio_stats(folder_path: str, filter_list: Optional[list[str]]=None) -> tuple[float, float, list[float]]:
    """Calculate the total duration of all audio files in a folder."""
    total_duration = 0.0
    nr_files = 0
    durations = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filter_list and filename not in filter_list:
                continue
            if filename.endswith('.wav'):
                nr_files += 1
                file_path = os.path.join(root, filename)
                total_duration += get_audio_duration(file_path)
                durations.append(get_audio_duration(file_path))
    avg_duration = total_duration / nr_files
    return total_duration, avg_duration, durations


# GENERAL HELPERS
def time_to_seconds(time_str: str) -> float:
    """Convert time in HH:MM:SS,SSS format to total seconds."""
    hours, minutes, seconds = time_str.split(':')
    seconds, millis = seconds.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * \
        60 + int(seconds) + int(millis) / 1000
    return total_seconds


def write_tokens_to_file(tokens: list[str], output_file: str) -> str:
    """Write a list of tokens to a text file, one token per line."""
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
    tokens = re.split(r'[ .,!?\[\]{}<>:;"\'()]+', text)
    return tokens


def basic_sentence_tokenizer(text: str) -> list[str]:
    """Basic sentence tokenization of a text string, considering punctuation and whitespace."""
    sentences = re.split(r'[.!?]', text)
    return sentences


def get_token_stats(directory: str):
    """Reads in all text files in a directory and calculates token statistics."""
    data = []
    ids = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            ident = file.split('_')[0] + '_' + file.split('_')[1]
            if ident in ids:
                raise ValueError(f'Duplicate ID: {ident}')
            ids.append(ident)
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                text = f.read().rstrip()
                tokens = basic_tokenizer(text)
                sentences = basic_sentence_tokenizer(text)
                nr_tokens = len(tokens)
                average_sent_length = nr_tokens / len(sentences)
                data.append({
                    'id': ident,
                    'text': text,
                    'nr_token': len(tokens),
                    'nr_sentences': len(sentences),
                    'avg_tokens_per_sentence': average_sent_length
                })
    return pd.DataFrame(data)


def hist_plot(df: pd.DataFrame, col: str, title: str, xlabel: str, save_path: str = None):
    """Create a histogram plot of a given DataFrame using seaborn"""
    sns.set_theme(style='whitegrid')
    plot = sns.histplot(data=df, x=col, bins=30)
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    if save_path:
        plot.get_figure().savefig(save_path)
    return plot


def main():
    xml_file = "data_raw/Mundartkorpus/XML-CHMK_v2.1_corpus/100008_Rhyner-Freitag_Buech.xml"
    sents = extract_text_chmk(xml_file)
    print(sents)
    xml_dir = 'data_raw/Mundartkorpus/XML-CHMK_v2.1_corpus'
    outdir = 'data_prepared/mundartkorpus'
    process_chmk(xml_dir, outdir)
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

if __name__ == "__main__":

    main()
