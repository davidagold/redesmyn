extern crate proc_macro;

use std::collections::{HashMap, VecDeque};

use kw::dimensions;
use proc_macro::TokenStream;
use proc_macro2::{extra::DelimSpan, Ident, Span};
use quote::{quote, ToTokens};
use syn::{
    bracketed, parenthesized,
    parse::Parse,
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    token::{self, Bracket},
    Attribute, Data, DeriveInput, Expr, Fields, FieldsNamed, ItemFn, LitStr, Meta, MetaList,
    MetaNameValue, Path, PathSegment, Signature, Token, Type, Visibility,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum MacroError {
    #[error("Target is not a struct.")]
    NotStructError(),
    #[error("Failed to extract fields from target.")]
    BadFieldsError(),
}

fn get_struct_fields(input: &DeriveInput) -> Result<&FieldsNamed, MacroError> {
    let data_struct = match &input.data {
        Data::Struct(data) => data,
        _ => return Err(MacroError::NotStructError()),
    };
    match &data_struct.fields {
        Fields::Named(fields) => Ok(fields),
        _ => Err(MacroError::BadFieldsError()),
    }
}

fn get_expr_data_type(type_ident: &Ident) -> proc_macro2::TokenStream {
    match type_ident.to_string().as_str() {
        "i64" => quote!(polars::datatypes::DataType::Int64),
        "f64" => quote!(polars::datatypes::DataType::Float64),
        "String" => quote!(polars::datatypes::DataType::String),
        _ => todo!(),
    }
}

#[proc_macro_derive(Relation)]
pub fn derive_relation(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

    let mut exprs_schema_fields: Vec<proc_macro2::TokenStream> = Vec::new();

    let mut exprs_vec_init: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut exprs_vec_push: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut exprs_series: Vec<proc_macro2::TokenStream> = Vec::new();

    let fields = match get_struct_fields(&input) {
        Ok(fields) => fields,
        Err(_) => panic!(),
    };

    for field in &fields.named {
        let ident_field = match &field.ident {
            Some(id) => id,
            None => panic!(),
        };

        match &field.ty {
            Type::Path(type_path) => {
                let type_segment = type_path.path.segments.last().unwrap();
                let type_ident = &type_segment.ident;
                exprs_vec_init.push(quote! {
                    let mut #ident_field = Vec::<#type_ident>::new()
                });

                exprs_vec_push.push(quote! {
                    #ident_field.push(record.#ident_field)
                });

                let name = ident_field.to_string();
                exprs_series.push(quote! {
                    Series::new(#name, #ident_field)
                });

                let expr_data_type = get_expr_data_type(type_ident);
                exprs_schema_fields.push(quote! {
                    .add_field(#name, #expr_data_type)
                });
            }
            _ => unimplemented!(),
        }
    }

    let gen = quote! {
        impl Relation for #name {
            type Serialized = String;

            fn schema(rel: Option<&Self>) -> Option<Schema> {
                let schema = redesmyn::schema::Schema::default()
                    #(#exprs_schema_fields)*;
                Some(schema)
            }

            fn parse(records: Vec<String>, schema: &redesmyn::schema::Schema) -> Result<polars::prelude::DataFrame, redesmyn::error::ServiceError>
            where
                Self: Sized
            {
                #(#exprs_vec_init);*;
                for record in records.iter().filter_map(|record| -> Option<#name> { serde_json::from_str(record).ok() }) {
                    #(#exprs_vec_push); *;
                };

                let columns: Vec<Series> = vec![#(#exprs_series), *];

                DataFrame::new(columns).map_err(|err| err.into())
            }
        }
    };

    gen.into()
}

#[derive(Default)]
struct MetricInstrumentArgs {
    dimensions: Vec<(PathSegment, Expr)>,
}

mod kw {
    use syn::custom_keyword;

    custom_keyword!(dimensions);
}

impl Parse for MetricInstrumentArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let lookahead = input.lookahead1();
            if lookahead.peek(kw::dimensions) {
                let _ = input.parse::<kw::dimensions>();
                let content;
                parenthesized!(content in input);
                for kw in content.parse_terminated(MetaNameValue::parse, Token![,])?.into_iter() {
                    let dim_name =
                        kw.path.segments.first().expect("Expected dimension name.").clone();
                    args.dimensions.push((dim_name, kw.value));
                }
            };
        }
        Ok(args)
    }
}

#[proc_macro_attribute]
pub fn metric_instrument(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MetricInstrumentArgs);
    let ItemFn { attrs, vis, sig, block } = parse_macro_input!(item as ItemFn);
    let Signature {
        output,
        inputs: params,
        unsafety,
        asyncness,
        constness,
        abi,
        ident,
        generics: syn::Generics { params: gen_params, where_clause, .. },
        ..
    } = sig;

    let dimensions: Vec<Expr> = args
        .dimensions
        .into_iter()
        .map(|(key, value)| parse_quote!(__Dimensions.#key = #value))
        .collect();

    quote!(
        #[tracing::instrument(skip_all, fields(#(#dimensions), *))]
        #(#attrs) *
        #vis #constness #unsafety #asyncness #abi fn #ident<#gen_params>(#params) #output
        #where_clause
        #block
    )
    .into()
}
