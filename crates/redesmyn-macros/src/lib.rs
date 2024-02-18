extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, FieldsNamed, Type};
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

#[proc_macro_derive(Record)]
pub fn to_dataframe_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

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
                })
            }
            _ => unimplemented!(),
        }
    }

    let gen = quote! {
        impl Record<#name> for #name {
            fn to_dataframe(records: Vec<#name>) -> PolarsResult<DataFrame> {
                #(#exprs_vec_init);*;

                for record in records {
                    #(#exprs_vec_push);*;
                };

                let columns: Vec<Series> = vec![
                    #(#exprs_series),*
                ];

                DataFrame::new(columns)
            }
        }
    };

    gen.into()
}
