import pytest
from sims.qa.generators.common import split_camel_case


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("ExampleWord", "example word"),
        ("CamelCaseExample", "camel case example"),
        ("SimpleTest", "simple test"),
        ("LowerCase", "lower case"),
        ("JSONParser", "json parser"),
        ("HTTPRequestHandler", "http request handler"),
        ("Single", "single"),
        ("AnotherOneBitesTheDust", "another one bites the dust"),
        ("", ""),
        ("alreadylowercase", "alreadylowercase"),
        ("XMLHttpRequest", "xml http request"),
        ("GPUImageProcessor", "gpu image processor"),
        ("ALLCAPS", "allcaps"),
        ("lowerUPPER", "lower upper"),
        ("TestWithNumbers123", "test with numbers 123"),
        ("123NumbersAtStart", "123 numbers at start"),
        ("Mixed123NumbersAndLetters", "mixed 123 numbers and letters"),
    ],
)
def test_split_camel_case(input_str, expected):
    assert split_camel_case(input_str) == expected
