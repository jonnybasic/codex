use super::*;
use crate::config::Config;
use crate::config::ConfigOverrides;
use codex_utils_absolute_path::AbsolutePathBuf;
use tempfile::tempdir;

#[tokio::test]
async fn test_build_provider_returns_none_for_air_gap() -> Result<(), Box<dyn Error>> {
    let temp_dir = tempdir()?;
    let codex_home = AbsolutePathBuf::from_absolute_path(temp_dir.path().to_path_buf()).unwrap();
    let config = Config::load_from_base_config_with_overrides(
        Default::default(),
        ConfigOverrides {
            cwd: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        },
        codex_home,
    )
    .await?;

    let provider = build_provider(&config, "0.160.0-airgap", None, false)?;
    assert!(
        provider.is_none(),
        "OtelProvider should be None for air-gap"
    );
    Ok(())
}

#[tokio::test]
async fn test_check_for_update_on_startup_defaults_to_false() -> Result<(), Box<dyn Error>> {
    let temp_dir = tempdir()?;
    let codex_home = AbsolutePathBuf::from_absolute_path(temp_dir.path().to_path_buf()).unwrap();
    let config = Config::load_from_base_config_with_overrides(
        Default::default(),
        ConfigOverrides {
            cwd: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        },
        codex_home,
    )
    .await?;

    assert!(
        !config.check_for_update_on_startup,
        "check_for_update_on_startup must default to false in air-gapped build"
    );
    Ok(())
}
