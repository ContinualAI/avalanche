import unittest

from avalanche.core import Agent


class SimpleAdaptableObject:
    def __init__(self):
        self.has_called_pre = False
        self.has_called_post = False

    def pre_adapt(self, agent, exp):
        self.has_called_pre = True

    def post_adapt(self, agent, exp):
        self.has_called_post = True


class AgentTests(unittest.TestCase):
    def test_adapt_hooks(self):
        agent = Agent()

        IS_FOO_PRE_CALLED = False

        def foo_pre(a, e):
            nonlocal IS_FOO_PRE_CALLED
            IS_FOO_PRE_CALLED = True

        agent.add_pre_hooks(foo_pre)

        IS_FOO_POST_CALLED = False

        def foo_post(a, e):
            nonlocal IS_FOO_POST_CALLED
            IS_FOO_POST_CALLED = True

        agent.add_post_hooks(foo_post)

        e = None
        agent.pre_adapt(e)
        assert IS_FOO_PRE_CALLED
        assert not IS_FOO_POST_CALLED

        IS_FOO_PRE_CALLED = False
        agent.post_adapt(e)
        assert IS_FOO_POST_CALLED
        assert not IS_FOO_PRE_CALLED

    def test_adapt_objects(self):
        agent = Agent()
        agent.obj = SimpleAdaptableObject()

        e = None
        agent.pre_adapt(e)
        assert agent.obj.has_called_pre
        assert not agent.obj.has_called_post

        agent.obj.has_called_pre = False
        agent.post_adapt(e)
        assert agent.obj.has_called_post
        assert not agent.obj.has_called_pre
