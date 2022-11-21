import os
import signal
import sys
import unittest
from subprocess import Popen
from typing import Union, Set
from unittest import TestSuite, TestCase

os.environ['DISTRIBUTED_TESTS'] = '1'


def get_distributed_test_cases(suite: Union[TestCase, TestSuite]) -> Set[str]:
    found_cases = set()
    if isinstance(suite, TestSuite):
        for x in suite:
            found_cases.update(get_distributed_test_cases(x))

    if isinstance(suite, TestCase):
        case_id = suite.id()

        if case_id.startswith('distributed.') or \
                case_id.startswith('tests.distributed.'):
            found_cases.add(case_id)

        if '_FailedTest' in case_id:
            raise RuntimeError(
                f'Errors encountered while listing test cases: {case_id}')

    return found_cases


def run_distributed_suites():
    cases_names = get_distributed_test_cases(
        unittest.defaultTestLoader.discover('.'))  # Don't change the path!
    cases_names = list(sorted(cases_names))
    print('Running', len(cases_names), 'tests')
    p = None
    success = True
    exited = False

    for case_name in cases_names:
        if exited:
            print('Exiting due to keyboard interrupt')
            break
        print('Running test:', case_name, flush=True)
        try:
            p = Popen(
                ['python', '-m', 'torch.distributed.run', '--nnodes=1',
                 '--nproc_per_node=4', '-m', 'unittest', case_name],
                stdout=sys.stdout, stderr=sys.stderr)
            p.communicate()
        except KeyboardInterrupt:
            success = False
            exited = True
            p.send_signal(signal.SIGINT)
        finally:
            exit_code = p.wait()
            print('Test completed with code', exit_code)
            success = success and exit_code == 0
            p = None

    if success:
        print('Tests completed successfully')
        exit(0)
    else:
        print('Tests terminated with errors')
        exit(1)


if __name__ == '__main__':
    run_distributed_suites()
