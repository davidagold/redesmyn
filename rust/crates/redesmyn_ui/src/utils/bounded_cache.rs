use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

#[derive(Debug)]
pub struct BoundedCache<K, V> {
    entries: HashMap<K, V>,
    lru_order: VecDeque<K>,
    capacity: usize,
}

impl<K, V> BoundedCache<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get_mut(key)
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.touch(&key);
            return;
        }

        self.entries.insert(key.clone(), value);
        self.lru_order.push_back(key);
        self.prune();
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(ix) = self.lru_order.iter().position(|candidate| candidate == key) {
            self.lru_order.remove(ix);
        }
        self.entries.remove(key)
    }

    pub fn retain(&mut self, mut f: impl FnMut(&K, &mut V) -> bool) {
        self.entries.retain(|k, v| f(k, v));
        self.lru_order.retain(|key| self.entries.contains_key(key));
        self.prune();
    }

    fn touch(&mut self, key: &K) {
        if let Some(ix) = self.lru_order.iter().position(|candidate| candidate == key) {
            self.lru_order.remove(ix);
        }
        self.lru_order.push_back(key.clone());
    }

    fn prune(&mut self) {
        while self.entries.len() > self.capacity {
            let Some(key) = self.lru_order.pop_front() else {
                break;
            };
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedCache;

    #[test]
    fn bounded_cache_prunes_lru() {
        let mut cache = BoundedCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);

        // Touch "a" so it becomes MRU; inserting "c" should prune "b".
        assert_eq!(cache.get(&"a"), Some(&1));
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(&3));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bounded_cache_get_mut_touches_lru() {
        let mut cache = BoundedCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);

        // Touch "a" so it becomes MRU; inserting "c" should prune "b".
        *cache.get_mut(&"a").expect("present") = 10;
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), Some(&10));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(&3));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bounded_cache_retain_removes_order_entries() {
        let mut cache = BoundedCache::new(10);
        cache.insert(1, "a");
        cache.insert(2, "b");
        cache.insert(3, "c");

        cache.retain(|key, _| *key != 2);

        assert_eq!(cache.get(&1), Some(&"a"));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some(&"c"));
        assert_eq!(cache.len(), 2);
    }
}
