import pytest

from sims.qa import question_templates, vsi_question_templates


@pytest.mark.parametrize(
    "template",
    [
        question_templates.OBJ_ABS_DISTANCE_TEMPLATE,
        vsi_question_templates.OBJ_ABS_DISTANCE_TEMPLATE,
    ],
)
def test_absolute_distance_template_defines_multiple_instance_rule(template):
    assert "pair of instances with the shortest distance" in template
