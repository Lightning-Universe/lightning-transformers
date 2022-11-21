# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.5] - 2022-11-21

### Fixed

- Fixed loading HF model ([#306](https://github.com/Lightning-AI/lightning-transformers/pull/306))
- Fixed passing config name to `CNNDailyMailSummarizationDataModule` ([#310](https://github.com/Lightning-AI/lightning-transformers/pull/310))
- Fixed move pipeline to `self.device` as default ([#309](https://github.com/Lightning-AI/lightning-transformers/pull/309))


## [0.2.4] - 2022-11-03

### Changed

- Added support for Lightning v1.8.0 ([#297](https://github.com/Lightning-AI/lightning-transformers/pull/297))


## [0.2.3] - 2022-10-08

### Changed

- Use `lightning-utilities` for compatibility ([#292](https://github.com/Lightning-AI/lightning-transformers/pull/292))


## [0.2.2] - 2022-10-07

### Changed

- Added support for Lightning v1.7.0 ([#284](https://github.com/Lightning-AI/lightning-transformers/pull/284))
- Allow generation, fix examples ([#271](https://github.com/Lightning-AI/lightning-transformers/pull/271))

### Fixed

- Fixed `LightningCLI` compatibility ([#288](https://github.com/Lightning-AI/lightning-transformers/pull/288))


## [0.2.1] - 2022-06-28

### Changed

- Simplified Large Model Support/Add Large Model Training  ([#269](https://github.com/Lightning-AI/lightning-transformers/pull/269))
- Refactored the code for model creation ([#268](https://github.com/Lightning-AI/lightning-transformers/pull/268))


## [0.2.0] - 2022-06-23

### Added

- Added big model support ([#263](https://github.com/Lightning-AI/lightning-transformers/pull/263))
- Added support for `Trainer.predict` method ([#261](https://github.com/Lightning-AI/lightning-transformers/pull/261))
- Added ViT Image Classification Support ([#252](https://github.com/Lightning-AI/lightning-transformers/pull/252))
- Allow streaming datasets for the language modeling task ([#256](https://github.com/Lightning-AI/lightning-transformers/pull/256))
- Micro to macro conversion for Classification Metrics ([#255](https://github.com/Lightning-AI/lightning-transformers/pull/255))
- Added ability to set kwargs for pipeline in load_from_checkpoint ([#204](https://github.com/Lightning-AI/lightning-transformers/pull/204))
- Added Masked Language Modeling ([#173](https://github.com/Lightning-AI/lightning-transformers/pull/173))
- Added more schedulers ([#143](https://github.com/Lightning-AI/lightning-transformers/pull/143))

### Changed

- Refactored for new class based approach ([#243](https://github.com/Lightning-AI/lightning-transformers/pull/243))
- Flatten inheritance ([#245](https://github.com/Lightning-AI/lightning-transformers/pull/245))
- Removed configs ([#264](https://github.com/Lightning-AI/lightning-transformers/pull/264))
- Removed config/hydra from repo ([#262](https://github.com/Lightning-AI/lightning-transformers/pull/262))
- Changed `gpus` to `devices` for compatability ([#239](https://github.com/Lightning-AI/lightning-transformers/pull/239))

### Fixed

- Fixed Token Classification ([#265](https://github.com/Lightning-AI/lightning-transformers/pull/265))
- Fixed broken bleu metrics ([#228](https://github.com/Lightning-AI/lightning-transformers/pull/228))
- Fixed a bug preventing a model evaluating on the test split ([#186](https://github.com/Lightning-AI/lightning-transformers/pull/186))
- Reflect default settings changed to using ddp ([#175](https://github.com/Lightning-AI/lightning-transformers/pull/175))


## [0.1.0] - 2022-04-21

- EVERYTHING
