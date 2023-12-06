"""Computes the CIDEr (Consensus-Based Image Description Evaluation) Metric."""

# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr
# (Consensus-Based Image Description Evaluation) Metric
# by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu>
# and Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer


class Cider:
  """Main Class to compute the CIDEr metric."""

  def __init__(self, n=4, sigma=6.0):
    # set cider to sum over 1 to 4-grams
    self._n = n
    # set the standard deviation parameter for gaussian penalty
    self._sigma = sigma

  def compute_score(self, gts, res):
    """Main function to compute CIDEr score.

    Args:
      gts: dictionary with key <image> and value <tokenized hypothesis /
        candidate sentence>
      res: dictionary with key <image> and value <tokenized reference sentence>

    Returns:
      Computed CIDEr float score for the corpus.
    """

    assert sorted(gts.keys()) == sorted(res.keys())
    imgids = list(gts.keys())

    cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

    # Sort the IDs to be able to have control over the order
    # of the individual scores.
    for iid in sorted(imgids):
      hypo = res[iid]
      ref = gts[iid]

      # Sanity check.
      assert isinstance(hypo, list)
      assert len(hypo) == 1
      assert isinstance(ref, list)
      assert ref

      cider_scorer += (hypo[0], ref)

    (score, scores) = cider_scorer.compute_score()

    return score, scores

  def method(self):
    return "CIDEr"
