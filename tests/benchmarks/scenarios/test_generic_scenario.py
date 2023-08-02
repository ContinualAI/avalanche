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
            lambda: CLExperience(5, None).train().current_experience,
        )

        self.assertRaises(
            MaskedAttributeError,
            lambda: CLExperience(5, None).eval().current_experience,
        )

        print("CURRENT_EXPERIENCE: ", CLExperience(5, None).current_experience)
        assert CLExperience(5, None).current_experience == 5


class StreamTests(unittest.TestCase):
    def test_stream_getitem(self):
        # streams should be indexable
        s = EagerCLStream(
            "a",
            [CLExperience(0, None), CLExperience(1, None), CLExperience(2, None)],
            None,
        )

        s[0]
        s[1]
        s[2]

        self.assertRaises(IndexError, lambda: s[3])

        # should be iterable
        for el in s:
            print(el)

    def test_stream_slicing(self):
        # streams should be sliceable
        s = EagerCLStream(
            "a",
            [CLExperience(0, None), CLExperience(1, None), CLExperience(2, None)],
            None,
        )

        ss = s[1:2]
        assert len(ss) == 1
        ss[0].current_experience

        ss = s[:2]
        assert len(ss) == 2
        ss = s[1:]
        assert len(ss) == 2

    def test_lazy_stream(self):
        # lazy streams should be iterable
        def ls():
            # Also tests if set_stream_info works correctly
            for el in [
                CLExperience(0, None),
                CLExperience(0, None),
                CLExperience(0, None),
            ]:
                yield el

        s = CLStream("a", ls(), None, set_stream_info=True)
        for i, el in enumerate(s):
            assert el.current_experience == i
            assert el.origin_stream == s


class ScenarioTests(unittest.TestCase):
    def test_scenario_streams(self):
        # streams should be indexable
        sa = EagerCLStream(
            "a",
            [CLExperience(1, None), CLExperience(2, None), CLExperience(3, None)],
            None,
        )
        sb = EagerCLStream("b", [CLExperience(12, None), CLExperience(13, None)], None)
        bench = CLScenario([sa, sb])

        bench.a_stream
        bench.b_stream


if __name__ == "__main__":
    unittest.main()
