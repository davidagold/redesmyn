extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Ident;
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

fn get_expr_data_type(type_ident: &Ident) -> proc_macro2::TokenStream {
    match type_ident.to_string().as_str() {
        "i64" => quote!(polars::datatypes::DataType::Int64),
        "f64" => quote!(polars::datatypes::DataType::Float64),
        "String" => quote!(polars::datatypes::DataType::String),
        _ => todo!()
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
                    schema.add_field(#name, #expr_data_type)
                });
            }
            _ => unimplemented!(),
        }
    }

    let gen = quote! {
        impl Relation for #name {
            fn schema<'r>(&self) -> Schema<'r> {
                let schema = schema::Schema::default()
                #(#exprs_schema_fields);*;
            }

            fn to_dataframe(records: Vec<&str>) -> PolarsResult<DataFrame> {
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
