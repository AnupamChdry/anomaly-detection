import pytest
from src.training.datamodule import get_datamodule, CATEGORIES

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("category", CATEGORIES)
def test_datamodule_loads(category):
    dm = get_datamodule(category)
    assert len(dm.train_data) > 0
    assert len(dm.test_data) > 0


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        get_datamodule("invalid_category")
