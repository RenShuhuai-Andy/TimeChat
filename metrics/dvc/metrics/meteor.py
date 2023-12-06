"""Python wrapper for METEOR implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import threading

import numpy as np
import six


class Meteor(object):
  """Meteor scorer."""

  def __init__(self,
               meteor_jar_path=None,
               java_jre_path=None,
               jdk_java_options=None):
    if java_jre_path:
      self.java_bin = java_jre_path
    elif 'JRE_BIN_JAVA' in os.environ:
      self.java_bin = os.environ['JRE_BIN_JAVA']
    else:
      self.java_bin = 'java'

    if meteor_jar_path:
      meteor_jar = meteor_jar_path
    else:
      meteor_jar = os.path.join(
          './metrics', 'meteor-1.5.jar'
      )

    assert os.path.exists(meteor_jar), meteor_jar

    jdk_java_options = jdk_java_options or ['-Xmx2G']
    meteor_cmd = [
        self.java_bin, '-jar', '-Xmx2G', meteor_jar, '-', '-', '-stdio',
        '-l', 'en', '-norm'
    ]

    self.meteor_p = subprocess.Popen(
        meteor_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.lock = threading.Lock()

  def compute_score(self, gts, res):
    """Compute METEOR scores."""
    with self.lock:
      assert sorted(gts.keys()) == sorted(res.keys())
      img_ids = sorted(gts.keys())
      scores = []

      eval_line = 'EVAL ||| '
      stats = self._stat(img_ids, res, gts)
      eval_line += ' ||| '.join(stats)
      self.meteor_p.stdin.write(six.ensure_binary(eval_line + '\n'))
      self.meteor_p.stdin.flush()
      scores = [float(six.ensure_str(self.meteor_p.stdout.readline()))
                for _ in img_ids]
      # get the aggregated value
      score = self.meteor_p.stdout.readline()
      # do not close the file inside this function to keep it open for full eval
    return float(score), np.asarray(scores)

  def method(self):
    return 'METEOR'

  def _stat(self, img_ids, hypothesis_str, reference_list):  # pylint: disable=missing-function-docstring
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    stat_lines = []
    for i in img_ids:
      assert len(hypothesis_str[i]) == 1
      hypo = hypothesis_str[i][0].replace('|||', '').replace('  ', ' ')
      score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list[i]),
                                 hypo))

      self.meteor_p.stdin.write(six.ensure_binary(score_line + '\n'))
      self.meteor_p.stdin.flush()
      stat_lines.append(six.ensure_str(self.meteor_p.stdout.readline()).strip())
    return stat_lines
