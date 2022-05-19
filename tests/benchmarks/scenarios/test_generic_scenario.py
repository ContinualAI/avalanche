import unittest

from avalanche.benchmarks.scenarios.generic_scenario import (
    CLExperience,
    EagerCLStream,
    CLScenario,
    CLStream,
    MaskedAttributeError,
)


class ExperienceTests(unittest.TestCase):
    def test_exp(self):
        # all experiences have a current_experience identifier
        # accessible only in logging mode.

        self.assertRaises(
            MaskedAttributeError,
            lambda: CLExperience(5).train().current_experience,
        )

        self.assertRaises(
            MaskedAttributeError,
            lambda: CLExperience(5).eval().current_experience,
        )

        print(
            "CURRENT_EXPERIENCE: ", CLExperience(5).logging().current_experience
        )
        assert CLExperience(5).logging().current_experience == 5


class StreamTests(unittest.TestCase):
    def test_stream_getitem(self):
        # streams should be indexable
        s = EagerCLStream("a", [CLExperience(), CLExperience(), CLExperience()])

        s[0]
        s[1]
        s[2]

        self.assertRaises(IndexError, lambda: s[3])

        # should be iterable
        for el in s:
            print(el)

    def test_stream_slicing(self):
        # streams should be sliceable
        s = EagerCLStream("a", [CLExperience(), CLExperience(), CLExperience()])

        ss = s[1:2]
        assert len(ss) == 1
        ss[0].current_experience

        ss = s[:2]
        ss = s[1:]

    def test_lazy_stream(self):
        # lazy streams should be iterable
        def ls():
            for el in [CLExperience(), CLExperience(), CLExperience()]:
                yield el

        s = CLStream("a", ls())
        for i, el in enumerate(s):
            assert el.current_experience == i


class ScenarioTests(unittest.TestCase):
    def test_scenario_streams(self):
        # streams should be indexable
        sa = EagerCLStream(
            "a", [CLExperience(1), CLExperience(2), CLExperience(3)]
        )
        sb = EagerCLStream("b", [CLExperience(12), CLExperience(13)])
        bench = CLScenario([sa, sb])

        bench.a_stream
        bench.b_stream
