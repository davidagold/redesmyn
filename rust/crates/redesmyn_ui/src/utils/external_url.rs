use gpui::App;

pub trait OpenExternalUrl {
    /// Returns `true` if the URL was opened.
    fn open_external_url(&mut self, url: &str) -> bool;
}

impl OpenExternalUrl for App {
    fn open_external_url(&mut self, url: &str) -> bool {
        let url = url.trim();
        if is_http_https_url(url) {
            self.open_url(url);
            true
        } else {
            false
        }
    }
}

pub fn is_http_https_url(url: &str) -> bool {
    let url = url.trim();
    starts_with_ignore_ascii_case(url, "http://") || starts_with_ignore_ascii_case(url, "https://")
}

fn starts_with_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    haystack
        .get(0..needle.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(needle))
}

