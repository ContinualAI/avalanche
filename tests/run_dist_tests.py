import os
from pathlib import Path
import signal
import sys
import unittest
from subprocess import Popen
from typing import Union, Set
from unittest import TestSuite, TestCase

import click


def get_distributed_test_cases(suite: Union[TestCase, TestSuite]) -> Set[str]:
    found_cases = set()
    if isinstance(suite, TestSuite):
        for x in suite:
            found_cases.update(get_distributed_test_cases(x))

    if isinstance(suite, TestCase):
        case_id = suite.id()

        if case_id.startswith("distributed.") or case_id.startswith(
            "tests.distributed."
        ):
            found_cases.add(case_id)

        if "_FailedTest" in case_id:
            raise RuntimeError(
                f"Errors encountered while listing test cases: {case_id}"
            )

    return found_cases


@click.command()
@click.argument("test_cases", nargs=-1)
def run_distributed_suites(test_cases):
    if Path.cwd().name != "tests":
        os.chdir(Path.cwd() / "tests")

    cases_names = get_distributed_test_cases(
        unittest.defaultTestLoader.discover(".")
    )  # Don't change the path!
    cases_names = list(sorted(cases_names))
    print(cases_names)
    if len(test_cases) > 0:
        test_cases = set(test_cases)
        cases_names = [x for x in cases_names if x in test_cases]

        if set(cases_names) != test_cases:
            print("Some cases have not been found!", test_cases - set(cases_names))
            sys.exit(1)

    print("Running", len(cases_names), "tests")
    p = None
    success = True
    exited = False
    failed_test_cases = set()

    use_gpu_in_tests = os.environ.get("USE_GPU", "false").lower() in ["1", "true"]
    if use_gpu_in_tests:
        print("Running tests using GPUs")
        import torch

        nproc_per_node = torch.cuda.device_count()
    else:
        print("Running tests using CPU only")
        nproc_per_node = 2

    for case_name in cases_names:
        if exited:
            print("Exiting due to keyboard interrupt")
            break
        print("Running test:", case_name, flush=True)
        try:
            my_env = os.environ.copy()
            my_env["DISTRIBUTED_TESTS"] = "1"
            p = Popen(
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    f"--nproc_per_node={nproc_per_node}",
                    "-m",
                    "unittest",
                    case_name,
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=my_env,
            )
            p.communicate()
        except KeyboardInterrupt:
            success = False
            exited = True
            p.send_signal(signal.SIGINT)
        finally:
            exit_code = p.wait()
            print("Test completed with code", exit_code)
            success = success and exit_code == 0
            p = None

            if exit_code != 0:
                failed_test_cases.add(case_name)

    if success:
        print("Tests completed successfully")
        sys.exit(0)
    else:
        print("The following tests terminated with errors:")
        for failed_case in sorted(failed_test_cases):
            print(failed_case)

        sys.exit(1)


if __name__ == "__main__":
    run_distributed_suites()
