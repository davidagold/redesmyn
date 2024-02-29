#[macro_export]
macro_rules! config_methods {
    ($($name:ident : $type:ty),*) => {
        $(
            fn $name(mut self, $name: $type) -> Result<Self, ServiceError> {
                let Ok(mut config) = self.get_config() else {
                    return Err(ServiceError::Error("Failed to retrieve config".to_string()))
                };
                config.$name = $name;
                self.set_config(Some(config));
                Ok(self)
            }
        )*
    }
}

pub(crate) type Sized128String = heapless::String<128>;
pub(crate) type Sized256String = heapless::String<256>;
