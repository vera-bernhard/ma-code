from data_raw.datareader import SRTReader, extract_audio_and_text_all, already_extracted
import os
import sys
import unittest
import shutil

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSRTProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_file = 'test/test_data/swiss_format.srt'
        self.test_file2 = 'test/test_data/swiss_format_idx_ano.srt'
        self.test_file3 = 'test/test_data/swiss_format_faulty_timestamp.srt'
        self.test_file4 = 'test/test_data/normal_format.srt'
        self.expected_output = [
            [0, 'E usfüerlichi Studiobeschriebig finde Sie uf SRF.ch Schrägstrich einer gegen hundert Studio. Alles am Stück und Zahle als Ziffere gschriibe.'],
            [1, 'D Angelique stoot vor dr Gegnerwand.'],
            [2, 'D Gabi stoot uf dr Quizinsle. Sie het kurzi bondi Hoor und treit e lachsfarbige Blazer, drunter e schwarzes Tshirt und e schwarzi Hose.'],
            [3, 'Sie treit e brunrote Lippestift und e grüeni Ketti us grosse Steiperle.'],
            [4, 'Sie lacht und falted d Händ über dr Brust.'],
            [5, 'D Leichter vonere grauhoorige mit Brülle und enere grau-Bruun Hoorige lüchte uf.']
        ]

        self.expected_output_faulty_data = [
            [0, 'E usfüerlichi Studiobeschriebig finde Sie uf SRF.ch Schrägstrich einer gegen hundert Studio. Alles am Stück und Zahle als Ziffere gschriibe.'],
            [1, 'D Angelique stoot vor dr Gegnerwand.'],
            [3, 'Sie treit e brunrote Lippestift und e grüeni Ketti us grosse Steiperle.'],
            [4, 'Sie lacht und falted d Händ über dr Brust.']
        ]
        self.audio = '/home/vera/Documents/Uni/Master/Master_Thesis/ma-code/data_raw/SRF_AD/audio_full_ad/1G100_509.wav'
        self.test_output_dir = './test_prepared'

    def test_srt_reader(self):
        """Test if SRTReader correctly parses Swiss format SRT files."""
        result = SRTReader(self.test_file)
        self.assertEqual(len(result.data), len(self.expected_output))
        for expected, sub in zip(self.expected_output, result):
            self.assertEqual(expected[0], sub.index)
            self.assertEqual(expected[1], sub.content)

    def test_srt_reader_with_anomaly(self):
        """Test SRTReader with an index anomaly."""
        result1 = SRTReader(self.test_file2)
        self.assertEqual(len(result1.data), len(self.expected_output))
        for expected, sub in zip(self.expected_output, result1):
            self.assertEqual(expected[0], sub.index)
            self.assertEqual(expected[1], sub.content)
    
    def test_srt_reader_with_faulty_timestamp(self):
        result2 = SRTReader(self.test_file3)
        self.assertEqual(len(result2.data), len(self.expected_output_faulty_data))
        for expected, sub in zip(self.expected_output_faulty_data, result2):
            print(expected, sub)
            self.assertEqual(expected[0], sub.index)
            self.assertEqual(expected[1], sub.content)

    def test_extract_audio_and_text_all(self):
        """Test extracting audio and text from a full SRT file."""
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

        list_audio_text = extract_audio_and_text_all(
            self.audio, self.test_file, self.test_output_dir, '99'
        )

        # Check number of files created
        expected_count = len(self.expected_output)
        self.assertEqual(len(list_audio_text), expected_count)

        # Verify text files
        for i, (_, text_file) in enumerate(list_audio_text):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            self.assertEqual(text, self.expected_output[i][1])

    def test_already_extracted(self):
        """Test if already_extracted correctly detects extracted files."""
        # Ensure directory exists
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

        # Create mock extracted files
        base_name = os.path.basename(self.audio).rstrip('.wav')
        for i in range(1, len(self.expected_output) + 1):
            with open(os.path.join(self.test_output_dir, f'{i}_{base_name}.wav'), 'w') as f:
                f.write("fake_audio_data")
            with open(os.path.join(self.test_output_dir, f'{i}_{base_name}.txt'), 'w') as f:
                f.write(self.expected_output[i - 1][1])

        self.assertTrue(already_extracted(
            self.test_file, base_name, self.test_output_dir))

    def test_srt_reader_normal_time_offset(self):
        """Test if SRTReader correctly parses normal format SRT files."""
        result = SRTReader(self.test_file4)
        self.assertEqual(len(result.data), len(self.expected_output[:-1]))
        for expected, sub in zip(self.expected_output, result):
            self.assertEqual(expected[0], sub.index)
            self.assertEqual(expected[1], sub.content)

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)


if __name__ == '__main__':
    unittest.main()
