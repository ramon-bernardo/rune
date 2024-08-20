//! This is the lossless and more relaxed parser for Rune.
//!
//! This produces a syntax tree that can be analysed using the provided methods.

mod classify;
pub(crate) use self::classify::{classify, NodeClass};

mod grammar;
pub(crate) use self::grammar::{object_key, ws};

mod parser;
use self::parser::Checkpoint;

mod tree;
pub(crate) use self::tree::{Ignore, MaybeNode, Node, NodeAt, NodeId, Remaining, Stream, Tree};

use crate::macros::TokenStream;
use crate::parse::Lexer;
use crate::{compile, SourceId};

use self::parser::{Parser, Source};

/// Prepare parsing of text input.
pub(crate) fn prepare_text(input: &str) -> Prepare<'_> {
    Prepare::new(Input::Text(input))
}

/// Prepare parsing of a token stream.
#[allow(unused)]
pub(crate) fn prepare_token_stream(token_stream: &TokenStream) -> Prepare<'_> {
    Prepare::new(Input::TokenStream(token_stream))
}

enum Input<'a> {
    Text(&'a str),
    TokenStream(&'a TokenStream),
}

/// A prepared parse.
pub(crate) struct Prepare<'a> {
    input: Input<'a>,
    without_processing: bool,
    ignore_whitespace: bool,
    shebang: bool,
    source_id: SourceId,
}

impl<'a> Prepare<'a> {
    fn new(input: Input<'a>) -> Self {
        Self {
            input,
            without_processing: false,
            ignore_whitespace: false,
            shebang: true,
            source_id: SourceId::new(0),
        }
    }

    /// Disable input processing.
    #[cfg(feature = "fmt")]
    pub(crate) fn without_processing(mut self) -> Self {
        self.without_processing = true;
        self
    }

    /// Configure whether to ignore whitespace.
    pub(crate) fn ignore_whitespace(mut self, ignore_whitespace: bool) -> Self {
        self.ignore_whitespace = ignore_whitespace;
        self
    }

    /// Configure a source id.
    #[allow(unused)]
    pub(crate) fn with_source_id(mut self, source_id: SourceId) -> Self {
        self.source_id = source_id;
        self
    }

    /// Parse the prepared input.
    pub(crate) fn parse(self) -> compile::Result<Tree> {
        let source = match self.input {
            Input::Text(source) => {
                let mut lexer = Lexer::new(source, self.source_id, self.shebang);

                if self.without_processing {
                    lexer = lexer.without_processing();
                }

                Source::lexer(lexer)
            }
            Input::TokenStream(token_stream) => Source::token_stream(token_stream.iter()),
        };

        let mut p = Parser::new(source);
        p.ignore_whitespace(self.ignore_whitespace);
        self::grammar::root(&mut p)?;
        p.build()
    }
}
