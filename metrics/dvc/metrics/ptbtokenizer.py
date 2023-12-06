"""PTBTokenizer."""

# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import subprocess
import tempfile

# pylint: disable=g-inconsistent-quotes

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
  """Python wrapper of Stanford PTBTokenizer."""

  def __init__(self,
               ptbtokenizer_jar_path=None,
               java_jre_path=None):
    if java_jre_path:
      self.java_bin = java_jre_path
    elif 'JRE_BIN_JAVA' in os.environ:
      self.java_bin = os.environ['JRE_BIN_JAVA']
    else:
      self.java_bin = 'java'

    if ptbtokenizer_jar_path:
      self.ptbtokenizer_jar = ptbtokenizer_jar_path
    else:
      self.ptbtokenizer_jar = os.path.join(
          "./metrics",
          "stanford-corenlp-3.4.1.jar",
      )

    assert os.path.exists(self.ptbtokenizer_jar), self.ptbtokenizer_jar

  def tokenize(self, captions_for_image):
    """Tokenization."""

    cmd = [self.java_bin, '-cp', self.ptbtokenizer_jar,
           'edu.stanford.nlp.process.PTBTokenizer',
           '-preserveLines', '-lowerCase']

    # ======================================================
    # prepare data for PTB Tokenizer
    # ======================================================
    final_tokenized_captions_for_image = {}
    image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]  # pylint: disable=g-complex-comprehension
    sentences = "\n".join(
        [  # pylint: disable=g-complex-comprehension
            c["caption"].replace("\n", " ")
            for k, v in captions_for_image.items()
            for c in v
        ]
    )

    # ======================================================
    # save sentences to temporary file
    # ======================================================
    fd, tmpfname = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as f:
      f.write(sentences)

    # ======================================================
    # tokenize sentence
    # ======================================================
    cmd.append(tmpfname)
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip().encode())[0]
    token_lines = token_lines.decode()
    lines = token_lines.split('\n')
    # remove temp file
    os.remove(tmpfname)

    # ======================================================
    # create dictionary for tokenized captions
    # ======================================================
    for k, line in zip(image_id, lines):
      if k not in final_tokenized_captions_for_image:
        final_tokenized_captions_for_image[k] = []
      tokenized_caption = ' '.join([w for w in line.rstrip().split(' ')
                                    if w not in PUNCTUATIONS])
      final_tokenized_captions_for_image[k].append(tokenized_caption)

    return final_tokenized_captions_for_image
