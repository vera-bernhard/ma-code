import os
import sys
import unittest

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_raw.datareader import parse_swiss_srt, read_srt_file, extract_audio_and_text_all


class TestParseSwissSRT(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_file = 'test_data/swiss_format.srt'
        self.test_file2 = 'test_data/swiss_format_idx_ano.srt'
        self.expected_output = [
            [1, 'E usfüerlichi Studiobeschriebig finde Sie uf SRF.ch Schrägstrich einer gegen hundert Studio. Alles am Stück und Zahle als Ziffere gschriibe.'],
            [2, 'D Angelique stoot vor dr Gegnerwand.'],
            [3, 'D Gabi stoot uf dr Quizinsle. Sie het kurzi bondi Hoor und treit e lachsfarbige Blazer, drunter e schwarzes Tshirt und e schwarzi Hose.'],
            [4, 'Sie treit e brunrote Lippestift und e grüeni Ketti us grosse Steiperle.'],
            [5, 'Sie lacht und falted d Händ über dr Brust.'],
            [6, 'D Leichter vonere grauhoorige mit Brülle und enere grau-Bruun Hoorige lüchte uf.'
             ]
        ]
        self.audio = '/home/vera/Documents/Uni/Master/Master_Thesis/ma-code/data_raw/SRF_AD/audio_full_ad/1G100_509.wav'

    def test_parse_swiss_srt(self):
        """Test if parse_swiss_srt correctly parses Swiss format SRT files."""
        with open(self.test_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            result = parse_swiss_srt(srt_content)

        for e, sub in zip(self.expected_output, result):
            self.assertEqual(e[0], sub.index)
            self.assertEqual(e[1], sub.content)
            
    def test_read_srt_file(self):
        srt_data = read_srt_file(self.test_file)
        for s, e in zip(srt_data, self.expected_output):
            self.assertEqual(s['idx'], e[0])
            self.assertEqual(s['text'], e[1])
    
    def test_read_srt_file_idx_ano(self):
        srt_data = read_srt_file(self.test_file2)
        for s, e in zip(srt_data, self.expected_output):
            self.assertEqual(s['idx'], e[0])
            self.assertEqual(s['text'], e[1])

    def test_extract_audio_and_text_all(self):
        test_dir = './test_prepared'
        list_audio_text = extract_audio_and_text_all(self.audio, self.test_file, test_dir, '99')

        # Check how many files were created
        nr = self.expected_output[-1][0]
        self.assertEqual(len(list_audio_text), nr)

        # iterate over the list of audio and text files
        for i, (audio_file, text_file) in enumerate(list_audio_text):
            # read in the text file
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().rstrip()
            # check if the text is correct
            self.assertEqual(text, self.expected_output[i][1])
    
    def tearDown(self):
        """Clean up test data."""
        test_dir = './test_prepared'
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)
        

if __name__ == '__main__':
    unittest.main()
