import builtins

from segment_detector_class import SegmentDetector
from ellipse_candidate_maker_class import EllipseCandidateMaker
from ellipse_estimator_class import EllipseEstimator
from ellipse_merger_class import EllipseMerger


class EllipseDetector(object):
    def __init__(self):
        pass

    def detect(self, image):
        """Detect ellipse from image.

        Args:
            image: A numpy as array indicats gray scale image.

        Returns:
            Array of Ellipse instance that was detected from image.
        """

        if len(image.shape) != 2:
            raise builtins.RuntimeError()

        seg_detector = SegmentDetector()
        segments = seg_detector.detect(image)

        ellipse_cand_maker = EllipseCandidateMaker()
        ellipse_cands = ellipse_cand_maker.make(segments)

        ellipse_estimator = EllipseEstimator()
        ellipses = ellipse_estimator.estimate(ellipse_cands)

        ellipse_merger = EllipseMerger(image.shape[1], image.shape[0])
        ellipses = ellipse_merger.merge(ellipses)

        return ellipses
