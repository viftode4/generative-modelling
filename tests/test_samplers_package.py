import pytest


def test_central_sampler_package_exports_core_implementations():
    from zigzag_project import samplers
    from zigzag_project.zsampling import ZSamplingConfig as LegacyZSamplingConfig

    assert samplers.ZSamplingConfig is LegacyZSamplingConfig
    assert callable(samplers.z_sample_sdxl)
    assert callable(samplers.z_sample_sd21)
    assert callable(samplers.resampling_sdxl)
    assert callable(samplers.standard_sdxl)


def test_sampler_registry_uses_method_specific_config_types():
    from zigzag_project.samplers import ResamplingConfig, StandardSamplingConfig, ZSamplingConfig
    from zigzag_project.samplers.registry import get_sampler

    assert get_sampler("standard").config_type is StandardSamplingConfig
    assert get_sampler("zsampling").config_type is ZSamplingConfig
    assert get_sampler("resampling").config_type is ResamplingConfig


def test_sampler_registry_rejects_cross_method_config_mismatch():
    from zigzag_project.samplers import ZSamplingConfig
    from zigzag_project.samplers.registry import validate_sampler_config

    with pytest.raises(TypeError, match="resampling.*ResamplingConfig"):
        validate_sampler_config("resampling", ZSamplingConfig())


def test_resampling_entrypoint_rejects_zsampling_config_before_pipeline_use():
    from zigzag_project.samplers import ZSamplingConfig, resampling_sdxl

    with pytest.raises(TypeError, match="resampling_sdxl config.*ResamplingConfig"):
        resampling_sdxl(object(), config=ZSamplingConfig())


def test_legacy_resampling_entrypoint_rejects_zsampling_config_before_pipeline_use():
    from zigzag_project.zsampling import ZSamplingConfig, resampling_sdxl

    with pytest.raises(TypeError, match="resampling_sdxl config.*ResamplingConfig"):
        resampling_sdxl(object(), config=ZSamplingConfig())
