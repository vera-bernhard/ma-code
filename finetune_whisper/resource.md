# Fine-tuning whisper

* Official Hugginface Whisper Fine-Tuning: https://huggingface.co/blog/fine-tune-whisper


### How to deal with a new language:
* https://github.com/openai/whisper/discussions/64#discussioncomment-5687125
    1. Remove the language prediction task so that Whisper doesn’t get caught up with the fact it’s working on a new language and just focuses on transcribing text as accurately as possible (set langauge=None, task=None in the tokenizer + processor)
    2. Keep the language prediction task and tell Whisper that the new language is the same as one of it’s current languages (e.g. if fine-tuning on Nepali, tell Whisper it’s actually predicting Hindi, since the two are linguistically most similar): our thinking here was that we’d be able to leverage Whisper’s knowledge of the most linguistically similar language to the new language that we were showing it (set langauge=nepalese, task=transcribe in the tokenizer + processor)

## How to deal with audio > 30s
https://colab.research.google.com/drive/1l290cRv4RdvuLNlSeo9WexByHaNWs3s3?usp=sharing#scrollTo=kj_mLS9EUhua
 https://huggingface.co/openai/whisper-tiny#long-form-transcription: