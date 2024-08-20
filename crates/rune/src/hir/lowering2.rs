use core::mem::{replace, take};

use num::ToPrimitive;
use tracing::instrument_ast;

use crate::alloc::prelude::*;
use crate::alloc::{self, HashMap, HashSet};
use crate::ast::{self, Kind, Span, Spanned};
use crate::compile::{meta, Error, ErrorKind, ItemId, Result, WithSpan};
use crate::grammar::{classify, object_key, Ignore, MaybeNode, Node, NodeClass, Remaining, Stream};
use crate::hash::ParametersBuilder;
use crate::hir;
use crate::parse::Resolve;
use crate::query::{self, GenericsParameters, Named2, Named2Kind, Used};
use crate::runtime::{ConstValue, Type, TypeCheck};
use crate::Hash;

use super::{Ctxt, Needs};

use Kind::*;

/// Lower a bare function.
#[instrument_ast(span = p)]
pub(crate) fn bare<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ItemFn<'hir>> {
    let body = statements(cx, None, p)?;

    Ok(hir::ItemFn {
        span: p.span(),
        args: &[],
        body,
    })
}

/// Lower a function item.
#[instrument_ast(span = p)]
pub(crate) fn item_fn<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    is_instance: bool,
) -> Result<hir::ItemFn<'hir>> {
    alloc_with!(cx, p);

    p.remaining(cx, Attribute)?.ignore(cx)?;
    p.eat(Modifiers);
    p.expect(K![fn])?;
    p.ast::<ast::Ident>()?;

    let mut args = Vec::new();

    p.expect(FnArgs)?.parse(|p| {
        p.expect(K!['('])?;

        let mut comma = Remaining::default();

        while let MaybeNode::Some(pat) = p.eat(Pat) {
            comma.exactly_one(cx)?;
            let pat = pat.parse(|p| self::pat_with(cx, p, is_instance))?;
            args.try_push(hir::FnArg::Pat(alloc!(pat)))?;
            comma = p.one(K![,]);
        }

        comma.at_most_one(cx)?;
        p.expect(K![')'])?;
        Ok(())
    })?;

    let body = p.expect(Block)?.parse(|p| block(cx, None, p))?;

    Ok(hir::ItemFn {
        span: p.span(),
        args: iter!(args),
        body,
    })
}

/// Lower a block.
#[instrument_ast(span = p)]
pub(crate) fn block<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    label: Option<&ast::Label>,
    p: &mut Stream<'_>,
) -> Result<hir::Block<'hir>> {
    p.expect(K!['{'])?;
    let block = p.expect(BlockBody)?.parse(|p| statements(cx, label, p))?;
    p.expect(K!['}'])?;
    Ok(block)
}

#[instrument_ast(span = p)]
fn statements<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    label: Option<&ast::Label>,
    p: &mut Stream<'_>,
) -> Result<hir::Block<'hir>> {
    alloc_with!(cx, p);

    let label = match label {
        Some(label) => Some(alloc_str!(label.resolve(resolve_context!(cx.q))?)),
        None => None,
    };

    cx.scopes.push(label)?;

    let at = cx.statements.len();

    let mut must_be_last = None;

    while let Some(node) = p.next() {
        let (needs_semi, class) = classify(&node);
        let span = node.span();

        match node.kind() {
            Local => {
                let stmt = hir::Stmt::Local(alloc!(node.parse(|p| local(cx, p))?));
                cx.statements.try_push(stmt)?;
            }
            Expr => {
                let expr = node.parse(|p| expr(cx, p))?;
                let stmt = hir::Stmt::Expr(&*alloc!(expr));
                cx.statements.try_push(stmt)?;
            }
            Item => {
                let semi = p.remaining(cx, K![;])?;

                if needs_semi {
                    semi.exactly_one(cx)?;
                } else {
                    semi.at_most_one(cx)?;
                }

                continue;
            }
            _ => {
                cx.error(node.expected("an expression or local"))?;
                continue;
            }
        };

        let semis = p.remaining(cx, K![;])?;

        if let Some(span) = must_be_last {
            cx.error(Error::new(
                span,
                ErrorKind::ExpectedBlockSemiColon {
                    #[cfg(feature = "emit")]
                    followed_span: span,
                },
            ))?;
        } else if matches!(class, NodeClass::Expr) && semis.is_absent() {
            must_be_last = Some(span);
        }

        if needs_semi {
            if let Some(span) = semis.trailing() {
                cx.error(Error::msg(span, "unused semi-colons"))?;
            }

            semis.at_least_one(cx)?;
        } else if !matches!(class, NodeClass::Expr) {
            if let Some(span) = semis.span() {
                cx.error(Error::msg(span, "unused semi-colons"))?;
            }

            semis.ignore(cx)?;
        } else {
            semis.at_most_one(cx)?;
        }
    }

    let value = 'out: {
        if must_be_last.is_none() {
            break 'out None;
        }

        match cx.statements.pop() {
            Some(hir::Stmt::Expr(e)) => Some(e),
            Some(stmt) => {
                cx.statements.try_push(stmt).with_span(&*p)?;
                None
            }
            None => None,
        }
    };

    let statements = iter!(cx.statements.drain(at..));

    let layer = cx.scopes.pop().with_span(&*p)?;

    Ok(hir::Block {
        span: p.span(),
        label,
        statements,
        value,
        drop: iter!(layer.into_drop_order()),
    })
}

/// Lower a local.
#[instrument_ast(span = p)]
pub(crate) fn local<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::Local<'hir>> {
    // Note: expression needs to be assembled before pattern, otherwise the
    // expression will see declarations in the pattern.

    p.expect(K![let])?;
    let pat = p.expect(Pat)?;
    p.expect(K![=])?;
    let expr = p.expect(Expr)?;

    let expr = expr.parse(|p| self::expr(cx, p))?;
    let pat = pat.parse(|p| self::pat(cx, p))?;

    Ok(hir::Local {
        span: p.span(),
        pat,
        expr,
    })
}

/// Lower an expression.
#[instrument_ast(span = p)]
pub(crate) fn expr<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::Expr<'hir>> {
    alloc_with!(cx, ast);

    p.eat(Modifiers);

    let node = p.pump()?;
    let kind = node.parse(|p| expr_inner(cx, p))?;

    if let Some(label) = cx.label.take() {
        return Err(Error::msg(label, "labels are not supported for expression"));
    };

    Ok(hir::Expr {
        span: p.span(),
        kind,
    })
}

#[instrument_ast(span = p)]
fn expr_only<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Expr<'hir>> {
    let kind = expr_inner(cx, p)?;

    Ok(hir::Expr {
        span: p.span(),
        kind,
    })
}

#[instrument_ast(span = p)]
fn expr_inner<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    let in_path = take(&mut cx.in_path);

    match p.kind() {
        Block => expr_block(cx, p),
        ConstBlock(item) => expr_const_block(cx, p, item),
        AsyncBlock(item) => expr_async_block(cx, p, item),
        Closure(item) => expr_closure(cx, p, item),
        IndexedPath(..) => expr_path(cx, p, in_path),
        ExprArray => expr_array(cx, p),
        ExprTuple => expr_tuple(cx, p),
        ExprObject => expr_object(cx, p),
        ExprChain => expr_chain(cx, p),
        ExprBinary => expr_binary(cx, p),
        ExprLit => expr_lit(cx, p),
        ExprAssign => expr_assign(cx, p),
        ExprWhile => expr_while(cx, p),
        ExprLoop => expr_loop(cx, p),
        ExprFor => expr_for(cx, p),
        ExprRange => expr_range(cx, p),
        ExprRangeInclusive => expr_range_inclusive(cx, p),
        ExprRangeFrom => expr_range_from(cx, p),
        ExprRangeFull => expr_range_full(cx, p),
        ExprRangeTo => expr_range_to(cx, p),
        ExprRangeToInclusive => expr_range_to_inclusive(cx, p),
        _ => Err(p.expected(Expr)),
    }
}

/// Lower the given block expression.
#[instrument_ast(span = p)]
fn expr_block<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);
    Ok(hir::ExprKind::Block(alloc!(block(cx, None, p)?)))
}

/// Lower the given async block expression.
#[instrument_ast(span = p)]
fn expr_const_block<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    item: ItemId,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    if cx.const_eval {
        return Ok(hir::ExprKind::Block(alloc!(block(cx, None, p)?)));
    }

    let item = cx.q.item_for(item).with_span(&*p)?;
    let meta = cx.lookup_meta(&*p, item.item, GenericsParameters::default())?;

    let meta::Kind::Const = meta.kind else {
        return Err(Error::expected_meta(
            &*p,
            meta.info(cx.q.pool)?,
            "constant block",
        ));
    };

    p.ignore();
    Ok(hir::ExprKind::Const(meta.hash))
}

/// Lower the given async block expression.
#[instrument_ast(span = p)]
fn expr_async_block<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    item: ItemId,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    if cx.const_eval {
        return Err(Error::msg(
            &*p,
            "async blocks are not supported in constant contexts",
        ));
    };

    let item = cx.q.item_for(item).with_span(&*p)?;
    let meta = cx.lookup_meta(&*p, item.item, GenericsParameters::default())?;

    let meta::Kind::AsyncBlock { call, do_move, .. } = meta.kind else {
        return Err(Error::expected_meta(
            &*p,
            meta.info(cx.q.pool)?,
            "async block",
        ));
    };

    cx.scopes.push_captures()?;
    let block = alloc!(block(cx, None, p)?);
    let layer = cx.scopes.pop().with_span(&*p)?;

    cx.q.set_used(&meta.item_meta)?;

    let captures = &*iter!(layer.captures().map(|(_, id)| id));

    let Some(queue) = cx.secondary_builds.as_mut() else {
        return Err(Error::new(&*p, ErrorKind::AsyncBlockInConst));
    };

    queue.try_push(query::SecondaryBuildEntry {
        item_meta: meta.item_meta,
        build: query::SecondaryBuild::AsyncBlock(query::AsyncBlock {
            hir: alloc!(hir::AsyncBlock { block, captures }),
            call,
        }),
    })?;

    Ok(hir::ExprKind::AsyncBlock(alloc!(hir::ExprAsyncBlock {
        hash: meta.hash,
        do_move,
        captures,
    })))
}

/// Lower the given path.
#[instrument_ast(span = p)]
fn expr_path<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    in_path: bool,
) -> Result<hir::ExprKind<'hir>> {
    fn is_self(p: &Stream<'_>) -> bool {
        matches!(p.kinds(), Some([K![self]]))
    }

    fn try_as_ident(p: &Stream<'_>) -> Option<ast::Ident> {
        let [node] = p.nodes()?;
        node.ast().ok()
    }

    alloc_with!(cx, p);

    if is_self(p) {
        let Some((id, _)) = cx.scopes.get(hir::Name::SelfValue)? else {
            return Err(Error::new(&*p, ErrorKind::MissingSelf));
        };

        p.ignore();
        return Ok(hir::ExprKind::Variable(id));
    }

    if let Needs::Value = cx.needs {
        if let Some(name) = try_as_ident(p) {
            let name = alloc_str!(name.resolve(resolve_context!(cx.q))?);

            if let Some((name, _)) = cx.scopes.get(hir::Name::Str(name))? {
                p.ignore();
                return Ok(hir::ExprKind::Variable(name));
            }
        }
    }

    // Caller has indicated that if they can't have a variable, they do indeed
    // want to treat it as a path.
    if in_path {
        p.ignore();
        return Ok(hir::ExprKind::Path);
    }

    let named = cx.q.convert_path2(p)?;
    let parameters = generics_parameters(cx, &named)?;

    if let Some(meta) = cx.try_lookup_meta(&*p, named.item, &parameters)? {
        return expr_path_meta(cx, &meta, &*p);
    }

    if let (Needs::Value, Named2Kind::Ident(local)) = (cx.needs, named.kind) {
        let local = local.resolve(resolve_context!(cx.q))?;

        // light heuristics, treat it as a type error in case the first
        // character is uppercase.
        if !local.starts_with(char::is_uppercase) {
            return Err(Error::new(
                &*p,
                ErrorKind::MissingLocal {
                    name: Box::<str>::try_from(local)?,
                },
            ));
        }
    }

    let kind = if !parameters.parameters.is_empty() {
        ErrorKind::MissingItemParameters {
            item: cx.q.pool.item(named.item).try_to_owned()?,
            parameters: parameters.parameters,
        }
    } else {
        ErrorKind::MissingItem {
            item: cx.q.pool.item(named.item).try_to_owned()?,
        }
    };

    Err(Error::new(&*p, kind))
}

/// Lower the given array.
#[instrument_ast(span = p)]
fn expr_array<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K!['['])?;

    let mut items = Vec::new();
    let mut comma = Remaining::default();

    while let MaybeNode::Some(node) = p.eat(Expr) {
        comma.exactly_one(cx)?;
        items.try_push(node.parse(|p| expr(cx, p))?)?;
        comma = p.one(K![,]);
    }

    comma.at_most_one(cx)?;
    p.expect(K![']'])?;

    let seq = alloc!(hir::ExprSeq {
        items: iter!(items)
    });

    Ok(hir::ExprKind::Vec(seq))
}

/// Lower the given tuple.
#[instrument_ast(span = p)]
fn expr_tuple<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K!['('])?;

    let mut items = Vec::new();
    let mut comma = Remaining::default();

    while let MaybeNode::Some(node) = p.eat(Expr) {
        comma.exactly_one(cx)?;
        items.try_push(node.parse(|p| expr(cx, p))?)?;
        comma = p.one(K![,]);
    }

    if items.len() <= 1 {
        comma.exactly_one(cx)?;
    } else {
        comma.at_most_one(cx)?;
    }

    p.expect(K![')'])?;

    let seq = alloc!(hir::ExprSeq {
        items: iter!(items)
    });

    Ok(hir::ExprKind::Tuple(seq))
}

/// Lower the given tuple.
#[instrument_ast(span = p)]
fn expr_object<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let key = p.pump()?;

    let mut assignments = Vec::new();
    let mut comma = Remaining::default();
    let mut keys_dup = HashMap::new();

    p.expect(K!['{'])?;

    while matches!(p.peek(), object_key!()) {
        comma.exactly_one(cx)?;

        let (key_span, key) = match p.peek() {
            K![str] => {
                let lit = p.ast::<ast::LitStr>()?;
                let string = lit.resolve(resolve_context!(cx.q))?;
                (lit.span(), alloc_str!(string.as_ref()))
            }
            K![ident] => {
                let ident = p.ast::<ast::Ident>()?;
                let string = ident.resolve(resolve_context!(cx.q))?;
                (ident.span(), alloc_str!(string))
            }
            _ => {
                return Err(p.expected("object key"));
            }
        };

        let assign = if p.eat(K![:]).is_some() {
            p.expect(Expr)?.parse(|p| expr(cx, p))?
        } else {
            let Some((name, _)) = cx.scopes.get(hir::Name::Str(key))? else {
                return Err(Error::new(
                    key_span,
                    ErrorKind::MissingLocal {
                        name: key.try_to_string()?.try_into()?,
                    },
                ));
            };

            hir::Expr {
                span: key_span,
                kind: hir::ExprKind::Variable(name),
            }
        };

        if let Some(_existing) = keys_dup.try_insert(key, key_span)? {
            return Err(Error::new(
                key_span,
                ErrorKind::DuplicateObjectKey {
                    #[cfg(feature = "emit")]
                    existing: _existing.span(),
                    #[cfg(feature = "emit")]
                    object: p.span(),
                },
            ));
        }

        assignments.try_push(hir::FieldAssign {
            key: (key_span, key),
            assign,
            position: None,
        })?;

        comma = p.one(K![,]);
    }

    comma.at_most_one(cx)?;
    p.expect(K!['}'])?;

    let mut check_object_fields = |fields: &HashMap<_, meta::FieldMeta>, item: &crate::Item| {
        let mut fields = fields.try_clone()?;

        for assign in assignments.iter_mut() {
            let Some(meta) = fields.remove(assign.key.1) else {
                return Err(Error::new(
                    assign.key.0,
                    ErrorKind::LitObjectNotField {
                        field: assign.key.1.try_into()?,
                        item: item.try_to_owned()?,
                    },
                ));
            };

            assign.position = Some(meta.position);
        }

        if let Some(field) = fields.into_keys().next() {
            return Err(Error::new(
                p.span(),
                ErrorKind::LitObjectMissingField {
                    field,
                    item: item.try_to_owned()?,
                },
            ));
        }

        Ok(())
    };

    let kind = match key.kind() {
        AnonymousObjectKey => hir::ExprObjectKind::Anonymous,
        IndexedPath(..) => {
            let (named, span) = key.parse(|p| Ok((cx.q.convert_path2(p)?, p.span())))?;
            let parameters = generics_parameters(cx, &named)?;
            let meta = cx.lookup_meta(&span, named.item, parameters)?;
            let item = cx.q.pool.item(meta.item_meta.item);

            match &meta.kind {
                meta::Kind::Struct {
                    fields: meta::Fields::Empty,
                    ..
                } => {
                    check_object_fields(&HashMap::new(), item)?;
                    hir::ExprObjectKind::EmptyStruct { hash: meta.hash }
                }
                meta::Kind::Struct {
                    fields: meta::Fields::Named(st),
                    constructor,
                    ..
                } => {
                    check_object_fields(&st.fields, item)?;

                    match constructor {
                        Some(_) => hir::ExprObjectKind::ExternalType {
                            hash: meta.hash,
                            args: st.fields.len(),
                        },
                        None => hir::ExprObjectKind::Struct { hash: meta.hash },
                    }
                }
                meta::Kind::Variant {
                    fields: meta::Fields::Named(st),
                    ..
                } => {
                    check_object_fields(&st.fields, item)?;
                    hir::ExprObjectKind::StructVariant { hash: meta.hash }
                }
                _ => {
                    return Err(Error::new(
                        span,
                        ErrorKind::UnsupportedLitObject {
                            meta: meta.info(cx.q.pool)?,
                        },
                    ));
                }
            }
        }
        _ => {
            return Err(p.expected("object key"));
        }
    };

    let object = alloc!(hir::ExprObject {
        kind,
        assignments: iter!(assignments),
    });

    Ok(hir::ExprKind::Object(object))
}

#[instrument_ast(span = p)]
fn expr_chain<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let node = p.pump()?;

    let in_path = replace(&mut cx.in_path, true);
    let label = cx.label.take();

    let (mut kind, mut span) = node.clone().parse(|p| Ok((expr_inner(cx, p)?, p.span())))?;
    let mut outer = Some(node);

    cx.in_path = in_path;
    cx.label = label;

    for node in p.by_ref() {
        let outer = outer.take();

        match node.kind() {
            ExprCall => {
                let span = replace(&mut span, node.span());
                kind = node.parse(|p| expr_call(cx, p, span, kind, outer))?;
            }
            ExprField => {
                let span = replace(&mut span, node.span());
                kind = node.parse(|p| expr_field(cx, p, span, kind))?;
            }
            ExprAwait => {
                let span = replace(&mut span, node.span());
                kind = node.parse(|p| expr_await(cx, p, span, kind))?;
            }
            _ => {
                return Err(node.expected(ExprChain));
            }
        }
    }

    Ok(kind)
}

#[instrument_ast(span = p)]
fn expr_binary<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let (mut lhs, mut lhs_span) = p.pump()?.parse(|p| Ok((expr_inner(cx, p)?, p.span())))?;

    while !p.is_eof() {
        let node = p.expect(ExprOperator)?;

        let Some(ops) = node.tokens::<2>() else {
            return Err(node.expected(ExprOperator));
        };

        let Some(op) = ast::BinOp::from_slice(&ops) else {
            return Err(node.expected(ExprOperator));
        };

        let rhs_needs = match op {
            ast::BinOp::As(..) | ast::BinOp::Is(..) | ast::BinOp::IsNot(..) => Needs::Type,
            _ => Needs::Value,
        };

        let needs = replace(&mut cx.needs, rhs_needs);
        let (rhs, rhs_span) = p.pump()?.parse(|p| Ok((expr_inner(cx, p)?, p.span())))?;
        cx.needs = needs;

        let span = lhs_span.join(rhs_span);
        let lhs_span = replace(&mut lhs_span, span);

        lhs = hir::ExprKind::Binary(alloc!(hir::ExprBinary {
            lhs: hir::Expr {
                span: lhs_span,
                kind: lhs
            },
            op,
            rhs: hir::Expr {
                span: rhs_span,
                kind: rhs
            },
        }));
    }

    Ok(lhs)
}

#[instrument_ast(span = p)]
fn expr_lit<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::ExprKind<'hir>> {
    let lit = lit(cx, p)?;
    Ok(hir::ExprKind::Lit(lit))
}

#[instrument_ast(span = p)]
fn expr_assign<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let lhs = p.expect(Expr)?.parse(|p| expr(cx, p))?;
    p.expect(K![=])?;
    let rhs = p.expect(Expr)?.parse(|p| expr(cx, p))?;

    Ok(hir::ExprKind::Assign(alloc!(hir::ExprAssign { lhs, rhs })))
}

#[instrument_ast(span = p)]
fn expr_while<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let label = match cx.label.take() {
        Some(label) => Some(alloc_str!(label.resolve(resolve_context!(cx.q))?)),
        None => None,
    };

    cx.scopes.push_loop(label)?;

    p.expect(K![while])?;
    let condition = p.pump()?.parse(|p| condition(cx, p))?;
    let body = p.expect(Block)?.parse(|p| block(cx, None, p))?;
    let layer = cx.scopes.pop().with_span(&*p)?;

    Ok(hir::ExprKind::Loop(alloc!(hir::ExprLoop {
        label,
        condition: Some(alloc!(condition)),
        body,
        drop: iter!(layer.into_drop_order()),
    })))
}

#[instrument_ast(span = p)]
fn expr_loop<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let label = match cx.label.take() {
        Some(label) => Some(alloc_str!(label.resolve(resolve_context!(cx.q))?)),
        None => None,
    };

    cx.scopes.push_loop(label)?;

    p.expect(K![loop])?;
    let body = p.expect(Block)?.parse(|p| block(cx, None, p))?;
    let layer = cx.scopes.pop().with_span(&*p)?;

    Ok(hir::ExprKind::Loop(alloc!(hir::ExprLoop {
        label,
        condition: None,
        body,
        drop: iter!(layer.into_drop_order()),
    })))
}

#[instrument_ast(span = p)]
fn expr_for<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K![for])?;
    let pat = p.expect(Pat)?;
    p.expect(K![in])?;
    let iter = p.expect(Expr)?;
    let block = p.expect(Block)?;

    let label = match cx.label.take() {
        Some(label) => Some(alloc_str!(label.resolve(resolve_context!(cx.q))?)),
        None => None,
    };

    let iter = iter.parse(|p| expr(cx, p))?;

    cx.scopes.push_loop(label)?;

    let binding = pat.parse(|p| self::pat(cx, p))?;
    let body = block.parse(|p| self::block(cx, None, p))?;

    let layer = cx.scopes.pop().with_span(&*p)?;

    Ok(hir::ExprKind::For(alloc!(hir::ExprFor {
        label,
        binding,
        iter,
        body,
        drop: iter!(layer.into_drop_order()),
    })))
}

#[instrument_ast(span = p)]
fn expr_range<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let start = p.pump()?.parse(|p| expr_only(cx, p))?;
    p.expect(K![..])?;
    let end = p.pump()?.parse(|p| expr_only(cx, p))?;

    Ok(hir::ExprKind::Range(alloc!(hir::ExprRange::Range {
        start,
        end,
    })))
}

#[instrument_ast(span = p)]
fn expr_range_inclusive<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let start = p.pump()?.parse(|p| expr_only(cx, p))?;
    p.expect(K![..=])?;
    let end = p.pump()?.parse(|p| expr_only(cx, p))?;

    Ok(hir::ExprKind::Range(alloc!(
        hir::ExprRange::RangeInclusive { start, end }
    )))
}

#[instrument_ast(span = p)]
fn expr_range_from<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let start = p.pump()?.parse(|p| expr_only(cx, p))?;
    p.expect(K![..])?;

    Ok(hir::ExprKind::Range(alloc!(hir::ExprRange::RangeFrom {
        start,
    })))
}

#[instrument_ast(span = p)]
fn expr_range_full<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);
    p.expect(K![..])?;
    Ok(hir::ExprKind::Range(alloc!(hir::ExprRange::RangeFull)))
}

#[instrument_ast(span = p)]
fn expr_range_to<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K![..])?;
    let end = p.pump()?.parse(|p| expr_only(cx, p))?;

    Ok(hir::ExprKind::Range(alloc!(hir::ExprRange::RangeTo {
        end,
    })))
}

#[instrument_ast(span = p)]
fn expr_range_to_inclusive<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K![..=])?;
    let end = p.pump()?.parse(|p| expr_only(cx, p))?;

    Ok(hir::ExprKind::Range(alloc!(
        hir::ExprRange::RangeToInclusive { end }
    )))
}

#[instrument_ast(span = p)]
fn condition<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
) -> Result<hir::Condition<'hir>> {
    alloc_with!(cx, p);

    Ok(match p.peek() {
        Condition => hir::Condition::ExprLet(alloc!(expr_let(cx, p)?)),
        _ => hir::Condition::Expr(alloc!(expr(cx, p)?)),
    })
}

#[instrument_ast(span = p)]
fn expr_let<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::ExprLet<'hir>> {
    p.expect(K![let])?;
    let pat = p.expect(Pat)?;
    p.expect(K![=])?;
    let expr = p.expect(Expr)?;

    let expr = expr.parse(|p| self::expr(cx, p))?;
    let pat = pat.parse(|p| self::pat(cx, p))?;

    Ok(hir::ExprLet { pat, expr })
}

/// Assemble a closure expression.
#[instrument_ast(span = p)]
fn expr_closure<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    item: ItemId,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    let Some(meta) = cx.q.query_meta(&*p, item, Used::default())? else {
        return Err(Error::new(
            &*p,
            ErrorKind::MissingItem {
                item: cx.q.pool.item(item).try_to_owned()?,
            },
        ));
    };

    let meta::Kind::Closure { call, do_move, .. } = meta.kind else {
        return Err(Error::expected_meta(
            &*p,
            meta.info(cx.q.pool)?,
            "a closure",
        ));
    };

    tracing::trace!("queuing closure build entry");

    cx.scopes.push_captures()?;

    let args = p.expect(ClosureArguments)?.parse(|p| {
        if matches!(p.peek(), K![||]) {
            p.pump()?;
            return Ok(&[][..]);
        };

        p.expect(K![|])?;

        let mut args = Vec::new();
        let mut comma = Remaining::default();

        while let MaybeNode::Some(pat) = p.eat(Pat) {
            comma.exactly_one(cx)?;
            let binding = pat.parse(|p| self::pat(cx, p))?;
            comma = p.remaining(cx, K![,])?;
            args.try_push(hir::FnArg::Pat(alloc!(binding)))
                .with_span(&*p)?;
        }

        comma.at_most_one(cx)?;
        p.expect(K![|])?;
        Ok(iter!(args))
    })?;

    let body = p.expect(Expr)?.parse(|p| expr(cx, p))?;
    let body = alloc!(body);

    let layer = cx.scopes.pop().with_span(&*p)?;

    cx.q.set_used(&meta.item_meta)?;

    let captures = &*iter!(layer.captures().map(|(_, id)| id));

    let Some(queue) = cx.secondary_builds.as_mut() else {
        return Err(Error::new(&*p, ErrorKind::ClosureInConst));
    };

    queue.try_push(query::SecondaryBuildEntry {
        item_meta: meta.item_meta,
        build: query::SecondaryBuild::Closure(query::Closure {
            hir: alloc!(hir::ExprClosure {
                args,
                body,
                captures,
            }),
            call,
        }),
    })?;

    if captures.is_empty() {
        return Ok(hir::ExprKind::Fn(meta.hash));
    }

    Ok(hir::ExprKind::CallClosure(alloc!(hir::ExprCallClosure {
        hash: meta.hash,
        do_move,
        captures,
    })))
}

#[instrument_ast(span = p)]
fn expr_call<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    span: Span,
    kind: hir::ExprKind<'hir>,
    outer: Option<Node<'_>>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K!['('])?;

    let mut comma = Remaining::default();
    let mut args = Vec::new();

    while let MaybeNode::Some(node) = p.eat(Expr) {
        comma.exactly_one(cx)?;
        let expr = node.parse(|p| expr(cx, p))?;
        args.try_push(expr)?;
        comma = p.one(K![,]);
    }

    comma.at_most_one(cx)?;
    p.expect(K![')'])?;

    let call = 'ok: {
        match kind {
            hir::ExprKind::Variable(name) => hir::Call::Var { name },
            hir::ExprKind::Path => {
                let Some(outer) = outer else {
                    return Err(Error::msg(span, "expected path"));
                };

                let (named, span) = outer.parse(|p| Ok((cx.q.convert_path2(p)?, p.span())))?;
                let parameters = generics_parameters(cx, &named)?;

                let meta = cx.lookup_meta(&span, named.item, parameters)?;
                debug_assert_eq!(meta.item_meta.item, named.item);

                match &meta.kind {
                    meta::Kind::Struct {
                        fields: meta::Fields::Empty,
                        ..
                    }
                    | meta::Kind::Variant {
                        fields: meta::Fields::Empty,
                        ..
                    } => {
                        if !args.is_empty() {
                            return Err(Error::new(
                                p,
                                ErrorKind::UnsupportedArgumentCount {
                                    expected: 0,
                                    actual: args.len(),
                                },
                            ));
                        }
                    }
                    meta::Kind::Struct {
                        fields: meta::Fields::Unnamed(expected),
                        ..
                    }
                    | meta::Kind::Variant {
                        fields: meta::Fields::Unnamed(expected),
                        ..
                    } => {
                        if *expected != args.len() {
                            return Err(Error::new(
                                p,
                                ErrorKind::UnsupportedArgumentCount {
                                    expected: *expected,
                                    actual: args.len(),
                                },
                            ));
                        }

                        if *expected == 0 {
                            cx.q.diagnostics.remove_tuple_call_parens(
                                cx.source_id,
                                p,
                                &span,
                                None,
                            )?;
                        }
                    }
                    meta::Kind::Function { .. } => {
                        if let Some(message) = cx.q.lookup_deprecation(meta.hash) {
                            cx.q.diagnostics.used_deprecated(
                                cx.source_id,
                                &span,
                                None,
                                message.try_into()?,
                            )?;
                        };
                    }
                    meta::Kind::ConstFn => {
                        let from = cx.q.item_for(named.item).with_span(span)?;

                        break 'ok hir::Call::ConstFn {
                            from_module: from.module,
                            from_item: from.item,
                            id: meta.item_meta.item,
                        };
                    }
                    _ => {
                        return Err(Error::expected_meta(
                            span,
                            meta.info(cx.q.pool)?,
                            "something that can be called as a function",
                        ));
                    }
                };

                hir::Call::Meta { hash: meta.hash }
            }
            hir::ExprKind::FieldAccess(&hir::ExprFieldAccess {
                expr_field,
                expr: target,
            }) => {
                let hash = match expr_field {
                    hir::ExprField::Index(index) => Hash::index(index),
                    hir::ExprField::Ident(ident) => {
                        cx.q.unit.insert_debug_ident(ident)?;
                        Hash::ident(ident)
                    }
                    hir::ExprField::IdentGenerics(ident, hash) => {
                        cx.q.unit.insert_debug_ident(ident)?;
                        Hash::ident(ident).with_function_parameters(hash)
                    }
                };

                hir::Call::Associated {
                    target: alloc!(target),
                    hash,
                }
            }
            _ => hir::Call::Expr {
                expr: alloc!(hir::Expr { span, kind }),
            },
        }
    };

    let kind = hir::ExprKind::Call(alloc!(hir::ExprCall {
        call,
        args: iter!(args),
    }));

    Ok(kind)
}

#[instrument_ast(span = p)]
fn expr_field<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    span: Span,
    kind: hir::ExprKind<'hir>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K![.])?;

    let expr_field = match p.peek() {
        K![number] => {
            let number = p.ast::<ast::LitNumber>()?;
            let index = number.resolve(resolve_context!(cx.q))?;

            let Some(index) = index.as_tuple_index() else {
                return Err(Error::new(
                    number,
                    ErrorKind::UnsupportedTupleIndex { number: index },
                ));
            };

            hir::ExprField::Index(index)
        }
        IndexedPath(..) => p.pump()?.parse(|p| match p.kinds() {
            Some([K![ident]]) => {
                let base = p.ast::<ast::Ident>()?;
                let base = base.resolve(resolve_context!(cx.q))?;
                let base = alloc_str!(base);
                Ok(hir::ExprField::Ident(base))
            }
            None => {
                let base = p.ast::<ast::Ident>()?;
                let base = base.resolve(resolve_context!(cx.q))?;
                let base = alloc_str!(base);

                if p.eat(K![::]).is_some() {
                    let hash = p
                        .expect(PathGenerics)?
                        .parse(|p| generic_arguments(cx, p))?;
                    Ok(hir::ExprField::IdentGenerics(base, hash))
                } else {
                    Ok(hir::ExprField::Ident(base))
                }
            }
            _ => Err(p.expected_peek(Path)),
        })?,
        _ => {
            return Err(p.expected(ExprField));
        }
    };

    let kind = hir::ExprKind::FieldAccess(alloc!(hir::ExprFieldAccess {
        expr: hir::Expr { span, kind },
        expr_field,
    }));

    Ok(kind)
}

#[instrument_ast(span = p)]
fn expr_await<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    span: Span,
    kind: hir::ExprKind<'hir>,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, p);

    p.expect(K![.])?;
    p.expect(K![await])?;

    Ok(hir::ExprKind::Await(alloc!(hir::Expr { span, kind })))
}

/// Compile an item.
#[instrument_ast(span = span)]
fn expr_path_meta<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    meta: &meta::Meta,
    span: &dyn Spanned,
) -> Result<hir::ExprKind<'hir>> {
    alloc_with!(cx, span);

    if let Needs::Value = cx.needs {
        match &meta.kind {
            meta::Kind::Struct {
                fields: meta::Fields::Empty,
                ..
            }
            | meta::Kind::Variant {
                fields: meta::Fields::Empty,
                ..
            } => Ok(hir::ExprKind::Call(alloc!(hir::ExprCall {
                call: hir::Call::Meta { hash: meta.hash },
                args: &[],
            }))),
            meta::Kind::Variant {
                fields: meta::Fields::Unnamed(0),
                ..
            }
            | meta::Kind::Struct {
                fields: meta::Fields::Unnamed(0),
                ..
            } => Ok(hir::ExprKind::Call(alloc!(hir::ExprCall {
                call: hir::Call::Meta { hash: meta.hash },
                args: &[],
            }))),
            meta::Kind::Struct {
                fields: meta::Fields::Unnamed(..),
                ..
            } => Ok(hir::ExprKind::Fn(meta.hash)),
            meta::Kind::Variant {
                fields: meta::Fields::Unnamed(..),
                ..
            } => Ok(hir::ExprKind::Fn(meta.hash)),
            meta::Kind::Function { .. } => Ok(hir::ExprKind::Fn(meta.hash)),
            meta::Kind::Const { .. } => Ok(hir::ExprKind::Const(meta.hash)),
            meta::Kind::Struct { .. } | meta::Kind::Type { .. } | meta::Kind::Enum { .. } => {
                Ok(hir::ExprKind::Type(Type::new(meta.hash)))
            }
            _ => Err(Error::expected_meta(
                span,
                meta.info(cx.q.pool)?,
                "something that can be used as a value",
            )),
        }
    } else {
        let Some(type_hash) = meta.type_hash_of() else {
            return Err(Error::expected_meta(
                span,
                meta.info(cx.q.pool)?,
                "something that has a type",
            ));
        };

        Ok(hir::ExprKind::Type(Type::new(type_hash)))
    }
}

fn pat<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::PatBinding<'hir>> {
    pat_with(cx, p, false)
}

fn pat_with<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    self_value: bool,
) -> Result<hir::PatBinding<'hir>> {
    alloc_with!(cx, p);
    let pat = p.pump()?.parse(|p| pat_only_with(cx, p, self_value))?;
    let names = iter!(cx.pattern_bindings.drain(..));
    Ok(hir::PatBinding { pat, names })
}

fn pat_only<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Pat<'hir>> {
    pat_only_with(cx, p, false)
}

fn pat_only_with<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    self_value: bool,
) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    match p.kind() {
        K![_] => Ok(hir::Pat {
            span: p.span(),
            kind: hir::PatKind::Ignore,
        }),
        IndexedPath(..) => pat_path(cx, p, self_value),
        PatLit => pat_lit(cx, p),
        PatTuple => pat_tuple(cx, p),
        PatObject => pat_object(cx, p),
        PatArray => pat_array(cx, p),
        _ => Err(p.expected(Pat)),
    }
}

#[instrument_ast(span = p)]
fn pat_path<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    p: &mut Stream<'_>,
    self_value: bool,
) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    let named = cx.q.convert_path2(p)?;
    let parameters = generics_parameters(cx, &named)?;

    let path = 'path: {
        if let Some(meta) = cx.try_lookup_meta(&*p, named.item, &parameters)? {
            match meta.kind {
                meta::Kind::Const => {
                    let Some(const_value) = cx.q.get_const_value(meta.hash) else {
                        return Err(Error::msg(
                            &*p,
                            try_format!("Missing constant for hash {}", meta.hash),
                        ));
                    };

                    let const_value = const_value.try_clone().with_span(&*p)?;
                    return pat_const_value(cx, &const_value, &*p);
                }
                _ => {
                    if let Some((0, kind)) = tuple_match_for(cx, &meta) {
                        break 'path hir::PatPathKind::Kind(alloc!(kind));
                    }
                }
            }
        };

        match named.kind {
            Named2Kind::SelfValue(ast) if self_value => {
                let name = cx.scopes.define(hir::Name::SelfValue, &ast)?;
                cx.pattern_bindings.try_push(name)?;
                break 'path hir::PatPathKind::Ident(name);
            }
            Named2Kind::Ident(ident) => {
                let name = alloc_str!(ident.resolve(resolve_context!(cx.q))?);
                let name = cx.scopes.define(hir::Name::Str(name), &*p)?;
                cx.pattern_bindings.try_push(name)?;
                break 'path hir::PatPathKind::Ident(name);
            }
            _ => {
                return Err(Error::new(&*p, ErrorKind::UnsupportedBinding));
            }
        }
    };

    let kind = hir::PatKind::Path(alloc!(path));

    Ok(hir::Pat {
        span: p.span(),
        kind,
    })
}

#[instrument_ast(span = p)]
fn pat_lit<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    let expr = alloc!(p.pump()?.parse(|p| expr_only(cx, p))?);

    Ok(hir::Pat {
        span: p.span(),
        kind: hir::PatKind::Lit(expr),
    })
}

#[instrument_ast(span = p)]
fn pat_tuple<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    let path = p.eat_matching(|kind| matches!(kind, IndexedPath(..)));

    p.expect(K!['('])?;

    let mut items = Vec::new();
    let mut comma = Remaining::default();

    while let MaybeNode::Some(pat) = p.eat(Pat) {
        comma.exactly_one(cx)?;
        items.try_push(pat.parse(|p| pat_only(cx, p))?)?;
        comma = p.one(K![,]);
    }

    let is_open = if p.eat(K![..]).is_some() {
        comma.exactly_one(cx)?;
        true
    } else {
        comma.at_most_one(cx)?;
        false
    };

    p.expect(K![')'])?;

    let items = iter!(items);

    let kind = if let MaybeNode::Some(path) = path {
        let (named, span) = path.parse(|p| Ok((cx.q.convert_path2(p)?, p.span())))?;
        let parameters = generics_parameters(cx, &named)?;
        let meta = cx.lookup_meta(&span, named.item, parameters)?;

        // Treat the current meta as a tuple and get the number of arguments it
        // should receive and the type check that applies to it.
        let Some((args, kind)) = tuple_match_for(cx, &meta) else {
            return Err(Error::expected_meta(
                span,
                meta.info(cx.q.pool)?,
                "type that can be used in a tuple pattern",
            ));
        };

        if !(args == items.len() || items.len() < args && is_open) {
            cx.error(Error::new(
                span,
                ErrorKind::UnsupportedArgumentCount {
                    expected: args,
                    actual: items.len(),
                },
            ))?;
        }

        kind
    } else {
        hir::PatSequenceKind::Anonymous {
            type_check: TypeCheck::Tuple,
            count: items.len(),
            is_open,
        }
    };

    Ok(hir::Pat {
        span: p.span(),
        kind: hir::PatKind::Sequence(alloc!(hir::PatSequence { kind, items })),
    })
}

#[instrument_ast(span = p)]
fn pat_object<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    let key = p.pump()?;

    let path = match key.kind() {
        AnonymousObjectKey => None,
        IndexedPath(..) => Some(key),
        _ => {
            return Err(p.expected_peek("object kind"));
        }
    };

    p.expect(K!['{'])?;

    let mut bindings = Vec::new();
    let mut comma = Remaining::default();
    let mut keys_dup = HashMap::new();

    while matches!(p.peek(), object_key!()) {
        comma.exactly_one(cx)?;

        let (span, key) = match p.peek() {
            K![str] => {
                let lit = p.ast::<ast::LitStr>()?;
                let string = lit.resolve(resolve_context!(cx.q))?;
                (lit.span(), alloc_str!(string.as_ref()))
            }
            K![ident] => {
                let ident = p.ast::<ast::Ident>()?;
                let string = ident.resolve(resolve_context!(cx.q))?;
                (ident.span(), alloc_str!(string))
            }
            _ => {
                return Err(p.expected_peek("object key"));
            }
        };

        if let Some(_existing) = keys_dup.try_insert(key, span)? {
            return Err(Error::new(
                span,
                ErrorKind::DuplicateObjectKey {
                    #[cfg(feature = "emit")]
                    existing: _existing.span(),
                    #[cfg(feature = "emit")]
                    object: p.span(),
                },
            ));
        }

        if p.eat(K![:]).is_some() {
            let pat = p.expect(Pat)?.parse(|p| pat_only(cx, p))?;
            bindings.try_push(hir::Binding::Binding(p.span(), key, alloc!(pat)))?;
        } else {
            let id = cx.scopes.define(hir::Name::Str(key), &*p)?;
            cx.pattern_bindings.try_push(id)?;
            bindings.try_push(hir::Binding::Ident(p.span(), key, id))?;
        }

        comma = p.one(K![,]);
    }

    let is_open = if p.eat(K![..]).is_some() {
        comma.exactly_one(cx)?;
        true
    } else {
        comma.at_most_one(cx)?;
        false
    };

    p.expect(K!['}'])?;

    let kind = match path {
        Some(path) => {
            let (named, span) = path.parse(|p| Ok((cx.q.convert_path2(p)?, p.span())))?;
            let parameters = generics_parameters(cx, &named)?;
            let meta = cx.lookup_meta(&span, named.item, parameters)?;

            let Some((mut fields, kind)) =
                struct_match_for(cx, &meta, is_open && bindings.is_empty())?
            else {
                return Err(Error::expected_meta(
                    span,
                    meta.info(cx.q.pool)?,
                    "type that can be used in a struct pattern",
                ));
            };

            for binding in bindings.iter() {
                if !fields.remove(binding.key()) {
                    return Err(Error::new(
                        span,
                        ErrorKind::LitObjectNotField {
                            field: binding.key().try_into()?,
                            item: cx.q.pool.item(meta.item_meta.item).try_to_owned()?,
                        },
                    ));
                }
            }

            if !is_open && !fields.is_empty() {
                let mut fields = fields.into_iter().try_collect::<Box<[_]>>()?;
                fields.sort();

                return Err(Error::new(
                    p.span(),
                    ErrorKind::PatternMissingFields {
                        item: cx.q.pool.item(meta.item_meta.item).try_to_owned()?,
                        #[cfg(feature = "emit")]
                        fields,
                    },
                ));
            }

            kind
        }
        None => hir::PatSequenceKind::Anonymous {
            type_check: TypeCheck::Object,
            count: bindings.len(),
            is_open,
        },
    };

    let bindings = iter!(bindings);

    Ok(hir::Pat {
        span: p.span(),
        kind: hir::PatKind::Object(alloc!(hir::PatObject { kind, bindings })),
    })
}

#[instrument_ast(span = p)]
fn pat_array<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, p);

    p.expect(K!['['])?;

    let mut items = Vec::new();
    let mut comma = Remaining::default();

    while let MaybeNode::Some(pat) = p.eat(Pat) {
        comma.exactly_one(cx)?;
        items.try_push(pat.parse(|p| pat_only(cx, p))?)?;
        comma = p.one(K![,]);
    }

    let is_open = if p.eat(K![..]).is_some() {
        comma.exactly_one(cx)?;
        true
    } else {
        comma.at_most_one(cx)?;
        false
    };

    p.expect(K![']'])?;

    let items = iter!(items);

    let kind = hir::PatSequenceKind::Anonymous {
        type_check: TypeCheck::Vec,
        count: items.len(),
        is_open,
    };

    Ok(hir::Pat {
        span: p.span(),
        kind: hir::PatKind::Sequence(alloc!(hir::PatSequence { kind, items })),
    })
}

fn generics_parameters(
    cx: &mut Ctxt<'_, '_, '_>,
    named: &Named2<'_>,
) -> Result<GenericsParameters> {
    let mut parameters = GenericsParameters {
        trailing: named.trailing,
        parameters: [None, None],
    };

    for (value, o) in named
        .parameters
        .iter()
        .zip(parameters.parameters.iter_mut())
    {
        if let Some(node) = value {
            let hash = node.clone().parse(|p| generic_arguments(cx, p))?;
            *o = Some(hash);
        }
    }

    Ok(parameters)
}

fn generic_arguments(cx: &mut Ctxt<'_, '_, '_>, p: &mut Stream<'_>) -> Result<Hash> {
    p.expect(K![<])?;

    let mut comma = Remaining::default();
    let mut builder = ParametersBuilder::new();

    let needs = replace(&mut cx.needs, Needs::Type);

    while matches!(p.peek(), IndexedPath(..)) {
        comma.exactly_one(cx)?;

        let hir::ExprKind::Type(ty) = p.pump()?.parse(|p| expr_path(cx, p, false))? else {
            return Err(Error::new(&*p, ErrorKind::UnsupportedGenerics));
        };

        builder.add(ty.into_hash());
        comma = p.one(K![,]);
    }

    cx.needs = needs;

    comma.at_most_one(cx)?;
    p.expect(K![>])?;
    Ok(builder.finish())
}

/// Construct a pattern from a constant value.
#[instrument_ast(span = span)]
fn pat_const_value<'hir>(
    cx: &mut Ctxt<'hir, '_, '_>,
    const_value: &ConstValue,
    span: &dyn Spanned,
) -> Result<hir::Pat<'hir>> {
    alloc_with!(cx, span);

    let kind = 'kind: {
        let lit = match *const_value {
            ConstValue::Bool(b) => hir::Lit::Bool(b),
            ConstValue::Byte(b) => hir::Lit::Byte(b),
            ConstValue::Char(ch) => hir::Lit::Char(ch),
            ConstValue::String(ref string) => hir::Lit::Str(alloc_str!(string.as_ref())),
            ConstValue::Bytes(ref bytes) => hir::Lit::ByteStr(alloc_bytes!(bytes.as_ref())),
            ConstValue::Integer(integer) => hir::Lit::Integer(integer),
            ConstValue::Vec(ref items) => {
                let items = iter!(items.iter(), items.len(), |value| pat_const_value(
                    cx, value, span
                )?);

                break 'kind hir::PatKind::Sequence(alloc!(hir::PatSequence {
                    kind: hir::PatSequenceKind::Anonymous {
                        type_check: TypeCheck::Vec,
                        count: items.len(),
                        is_open: false,
                    },
                    items,
                }));
            }
            ConstValue::Unit => {
                break 'kind hir::PatKind::Sequence(alloc!(hir::PatSequence {
                    kind: hir::PatSequenceKind::Anonymous {
                        type_check: TypeCheck::Unit,
                        count: 0,
                        is_open: false,
                    },
                    items: &[],
                }));
            }
            ConstValue::Tuple(ref items) => {
                let items = iter!(items.iter(), items.len(), |value| pat_const_value(
                    cx, value, span
                )?);

                break 'kind hir::PatKind::Sequence(alloc!(hir::PatSequence {
                    kind: hir::PatSequenceKind::Anonymous {
                        type_check: TypeCheck::Vec,
                        count: items.len(),
                        is_open: false,
                    },
                    items,
                }));
            }
            ConstValue::Object(ref fields) => {
                let bindings = iter!(fields.iter(), fields.len(), |(key, value)| {
                    let pat = alloc!(pat_const_value(cx, value, span)?);

                    hir::Binding::Binding(span.span(), alloc_str!(key.as_ref()), pat)
                });

                break 'kind hir::PatKind::Object(alloc!(hir::PatObject {
                    kind: hir::PatSequenceKind::Anonymous {
                        type_check: TypeCheck::Object,
                        count: bindings.len(),
                        is_open: false,
                    },
                    bindings,
                }));
            }
            _ => {
                return Err(Error::msg(span, "Unsupported constant value in pattern"));
            }
        };

        hir::PatKind::Lit(alloc!(hir::Expr {
            span: span.span(),
            kind: hir::ExprKind::Lit(lit),
        }))
    };

    Ok(hir::Pat {
        span: span.span(),
        kind,
    })
}

/// Generate a legal struct match for the given meta which indicates the type of
/// sequence and the fields that it expects.
///
/// For `open` matches (i.e. `{ .. }`), `Unnamed` and `Empty` structs are also
/// supported and they report empty fields.
fn struct_match_for(
    cx: &Ctxt<'_, '_, '_>,
    meta: &meta::Meta,
    open: bool,
) -> alloc::Result<Option<(HashSet<Box<str>>, hir::PatSequenceKind)>> {
    let (fields, kind) = match &meta.kind {
        meta::Kind::Struct { fields, .. } => {
            (fields, hir::PatSequenceKind::Type { hash: meta.hash })
        }
        meta::Kind::Variant {
            enum_hash,
            index,
            fields,
            ..
        } => {
            let kind = if let Some(type_check) = cx.q.context.type_check_for(meta.hash) {
                hir::PatSequenceKind::BuiltInVariant { type_check }
            } else {
                hir::PatSequenceKind::Variant {
                    variant_hash: meta.hash,
                    enum_hash: *enum_hash,
                    index: *index,
                }
            };

            (fields, kind)
        }
        _ => {
            return Ok(None);
        }
    };

    let fields = match fields {
        meta::Fields::Unnamed(0) if open => HashSet::new(),
        meta::Fields::Empty if open => HashSet::new(),
        meta::Fields::Named(st) => st
            .fields
            .keys()
            .try_cloned()
            .try_collect::<alloc::Result<_>>()??,
        _ => return Ok(None),
    };

    Ok(Some((fields, kind)))
}

fn tuple_match_for(
    cx: &Ctxt<'_, '_, '_>,
    meta: &meta::Meta,
) -> Option<(usize, hir::PatSequenceKind)> {
    Some(match &meta.kind {
        meta::Kind::Struct {
            fields: meta::Fields::Empty,
            ..
        } => (0, hir::PatSequenceKind::Type { hash: meta.hash }),
        meta::Kind::Struct {
            fields: meta::Fields::Unnamed(args),
            ..
        } => (*args, hir::PatSequenceKind::Type { hash: meta.hash }),
        meta::Kind::Variant {
            enum_hash,
            index,
            fields,
            ..
        } => {
            let args = match fields {
                meta::Fields::Unnamed(args) => *args,
                meta::Fields::Empty => 0,
                _ => return None,
            };

            let kind = if let Some(type_check) = cx.q.context.type_check_for(meta.hash) {
                hir::PatSequenceKind::BuiltInVariant { type_check }
            } else {
                hir::PatSequenceKind::Variant {
                    variant_hash: meta.hash,
                    enum_hash: *enum_hash,
                    index: *index,
                }
            };

            (args, kind)
        }
        _ => return None,
    })
}

#[instrument_ast(span = p)]
fn lit<'hir>(cx: &mut Ctxt<'hir, '_, '_>, p: &mut Stream<'_>) -> Result<hir::Lit<'hir>> {
    alloc_with!(cx, p);

    let node = p.pump()?;

    match node.kind() {
        K![true] => Ok(hir::Lit::Bool(true)),
        K![false] => Ok(hir::Lit::Bool(false)),
        K![number] => {
            let lit = node.ast::<ast::LitNumber>()?;
            let n = lit.resolve(resolve_context!(cx.q))?;

            match (n.value, n.suffix) {
                (ast::NumberValue::Float(n), _) => Ok(hir::Lit::Float(n)),
                (ast::NumberValue::Integer(int), Some(ast::NumberSuffix::Byte(..))) => {
                    let Some(n) = int.to_u8() else {
                        return Err(Error::new(lit, ErrorKind::BadNumberOutOfBounds));
                    };

                    Ok(hir::Lit::Byte(n))
                }
                (ast::NumberValue::Integer(int), _) => {
                    let Some(n) = int.to_i64() else {
                        return Err(Error::new(lit, ErrorKind::BadNumberOutOfBounds));
                    };

                    Ok(hir::Lit::Integer(n))
                }
            }
        }
        K![byte] => {
            let lit = node.ast::<ast::LitByte>()?;
            let b = lit.resolve(resolve_context!(cx.q))?;
            Ok(hir::Lit::Byte(b))
        }
        K![char] => {
            let lit = node.ast::<ast::LitChar>()?;
            let ch = lit.resolve(resolve_context!(cx.q))?;
            Ok(hir::Lit::Char(ch))
        }
        K![str] => {
            let lit = node.ast::<ast::LitStr>()?;

            let string = if cx.in_template {
                lit.resolve_template_string(resolve_context!(cx.q))?
            } else {
                lit.resolve_string(resolve_context!(cx.q))?
            };

            Ok(hir::Lit::Str(alloc_str!(string.as_ref())))
        }
        K![bytestr] => {
            let lit = node.ast::<ast::LitByteStr>()?;
            let bytes = lit.resolve(resolve_context!(cx.q))?;
            Ok(hir::Lit::ByteStr(alloc_bytes!(bytes.as_ref())))
        }
        _ => Err(node.expected(ExprLit)),
    }
}
