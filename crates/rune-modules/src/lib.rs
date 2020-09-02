//! <div align="center">
//! <a href="https://rune-rs.github.io/rune/">
//!     <b>Read the Book 📖</b>
//! </a>
//! </div>
//!
//! <br>
//!
//! <div align="center">
//! <a href="https://github.com/rune-rs/rune/actions">
//!     <img alt="Build Status" src="https://github.com/rune-rs/rune/workflows/Build/badge.svg">
//! </a>
//!
//! <a href="https://github.com/rune-rs/rune/actions">
//!     <img alt="Book Status" src="https://github.com/rune-rs/rune/workflows/Book/badge.svg">
//! </a>
//!
//! <a href="https://discord.gg/v5AeNkT">
//!     <img alt="Chat on Discord" src="https://img.shields.io/discord/558644981137670144.svg?logo=discord&style=flat-square">
//! </a>
//! </div>
//!
//! Native modules for the runestick virtual machine.
//!
//! These are modules that can be used with the [Rune language].
//!
//! [Rune Language]: https://github.com/rune-rs/rune
//!
//! See each module for documentation:
//! * [http](https://docs.rs/rune-modules/rune_modules/http/)
//! * [json](https://docs.rs/rune-modules/rune_modules/json/)
//! * [time](https://docs.rs/rune-modules/rune_modules/time/)
//! * [process](https://docs.rs/rune-modules/rune_modules/process/)
//!
//! ## Features
//! The `full` feature includes all modules.
//!
//! Apart from this, you can select which features to build:
//!
//! * [http](https://docs.rs/rune-modules/rune_modules/http/)
//! * [json](https://docs.rs/rune-modules/rune_modules/json/)
//! * [time](https://docs.rs/rune-modules/rune_modules/time/)
//! * [process](https://docs.rs/rune-modules/rune_modules/process/)

#[cfg(feature = "http")]
pub mod http;

#[cfg(feature = "json")]
pub mod json;

#[cfg(feature = "time")]
pub mod time;

#[cfg(feature = "process")]
pub mod process;
