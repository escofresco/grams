#!python

from .linkedlist import LinkedList

__all__ = []


class HashTable(object):
    __slots__ = ("buckets", "table_size", "load_factor")

    def __init__(self, init_size=8):
        """Initialize this hash table with the given initial size."""
        # Create a new list (used as fixed-size array) of empty linked lists
        self.buckets = [LinkedList() for _ in range(init_size)]
        self.table_size = 0
        self.load_factor = .9

    def __str__(self):
        """Return a formatted string representation of this hash table."""
        items = ['{!r}: {!r}'.format(key, val) for key, val in self.items()]
        return '{' + ', '.join(items) + '}'

    def __repr__(self):
        """Return a string representation of this hash table."""
        return 'HashTable({!r})'.format(self.items())

    def __iter__(self):
        yield from self.items()

    def _bucket_index(self, key):
        """Return the bucket index where the given key would be stored."""
        # Calculate the given key's hash code and transform into bucket index
        return hash(key) % len(self.buckets)

    def keys(self):
        """Return a list of all keys in this hash table.
         time:  Θ(n) for the best and worst case; there's never a condition
                when this function doesn't go through every item."""
        # Collect all keys in each bucket
        all_keys = []
        for bucket in self.buckets:
            for key, value in bucket.items():
                all_keys.append(key)
        return all_keys

    def values(self):
        """Return a list of all values in this hash table.
        time:  Θ(n) for the best and worst case; there's never a condition
               when this function doesn't go through every item."""
        return [value for ll in self.buckets for _, value in ll.items()]

    def items(self):
        """Return a list of all items (key-value pairs) in this hash table.
        time:  Θ(n) for the best and worst case; there's never a condition
               when this function doesn't go through every item."""
        # Collect all pairs of key-value entries in each bucket
        all_items = []
        for bucket in self.buckets:
            all_items.extend(bucket.items())
        return all_items

    def length(self):
        """Return the number of key-value entries by traversing its buckets.
        time:   Θ(1) for the best and worst case; length is tracked when other
                methods are called."""
        return sum(ll.length() for ll in self.buckets)

    def contains(self, key):
        """Return True if this hash table contains the given key, or False.
        time:   O(n/b) in the worst case; the bucket's entire list is looked
                through.
                Θ(1) in the best case; the bucket's list is either empty or
                the key matches the first node."""
        try:
            self.get(key)
        except:
            return False
        return True

    def get(self, key):
        """Return the value associated with the given key, or raise KeyError.
        time:   O(n/b) in the worst case; the bucket's entire list is looked
                through.
                Θ(1) in the best case; the bucket's list is either empty or
                the key matches the first node."""
        bucket_idx = self._bucket_index(key)
        val = self.buckets[bucket_idx].find(lambda item: item[0] == key)

        if val is None:
            raise KeyError(f"Key not found: {key}")
        return val[1]

    def set(self, key, value):
        """Insert or update the given key with its associated value.
        time:   O(n/b) in the worst case; the bucket's entire list is looked
                through.
                Θ(1) in the best case; the bucket's list is either empty or
                the key matches the first node."""
        bucket_idx = self._bucket_index(key)

        # subtract this linked list's length from our table
        self.table_size -= self.buckets[bucket_idx].length()

        self.buckets[bucket_idx].replace(lambda item: item[0] == key,
                                         (key, value))

        # add back this linked list's length (the differential being 1 or 0)
        self.table_size += self.buckets[bucket_idx].length()

    def delete(self, key):
        """Delete the given key from this hash table, or raise KeyError.
        time:   O(n/b) in the worst case; the bucket's entire list is looked
                through.
                Θ(1) in the best case; the bucket's list is either empty or
                the key matches the first node."""
        bucket_idx = self._bucket_index(key)

        try:
            self.buckets[bucket_idx].replace(lambda item: item[0] == key)
            self.table_size -= 1
        except KeyError as e:
            raise e

    @property
    def load(self):
        """Give the current load of the table"""
        return self.table_size / len(self.buckets)


def test_hash_table():
    ht = HashTable()
    print('hash table: {}'.format(ht))

    print('\nTesting set:')
    for key, value in [('I', 1), ('V', 5), ('X', 10)]:
        print('set({!r}, {!r})'.format(key, value))
        ht.set(key, value)
        print('hash table: {}'.format(ht))

    print('\nTesting get:')
    for key in ['I', 'V', 'X']:
        value = ht.get(key)
        print('get({!r}): {!r}'.format(key, value))

    print('contains({!r}): {}'.format('X', ht.contains('X')))
    print('length: {}'.format(ht.length()))

    # Enable this after implementing delete method
    delete_implemented = False
    if delete_implemented:
        print('\nTesting delete:')
        for key in ['I', 'V', 'X']:
            print('delete({!r})'.format(key))
            ht.delete(key)
            print('hash table: {}'.format(ht))

        print('contains(X): {}'.format(ht.contains('X')))
        print('length: {}'.format(ht.length()))


if __name__ == '__main__':
    test_hash_table()
