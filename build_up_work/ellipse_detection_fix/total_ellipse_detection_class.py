# coding: utf-8
from ellipse_class import Ellipse
from segment_class import Segment
from segment_pair_class import SegmentPair
from ellipse_candidate_class import EllipseCandidate
from segment_detector_class import SegmentDetector
from ellipse_center_estimator_class import EllipseCenterEstimator
from ellipse_candidate_maker_class import EllipseCandidateMaker
from ellipse_estimator_class import EllipseEstimator
from ellipse_merger_class import EllipseMerger
from ellipse_detector_class import EllipseDetector

class TotalEllipseDetector():    
    def __init__(self):
        self.ellipse = Ellipse()
        self.segment = Segment()
        self.segment_pair = SegmentPair()
        self.ellipse_candidate = EllipseCandidate()
        self.segment_detector = SegmentDetector()
        self.ellipse_center_estimator = EllipseCenterEstimator()
        self.ellipse_candidate_maker = EllipseCandidateMaker()
        self.ellipse_estimator = EllipseEstimator()
        self.ellipse_merger = EllipseMerger()
        self.ellipse_detector = EllipseDetector()
