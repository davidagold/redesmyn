#[macro_export]
macro_rules! config_methods {
    ($($name:ident : $type:ty),*) => {
        $(
            fn $name(mut self, $name: $type) -> Self {
                let mut config = self.config(None);
                config.$name = $name;
                self.config(Some(config));
                self
            }
        )*
    }
}
