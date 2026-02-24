use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::Node;

/// Number of nodes per chunk. Balances allocation frequency vs memory waste.
const CHUNK_SIZE: usize = 1024;

/// Maximum number of chunks. Supports up to MAX_CHUNKS * CHUNK_SIZE nodes.
/// 262144 * 1024 = ~268 million nodes. Uses 2MB for the pointer table.
const MAX_CHUNKS: usize = 262_144;

/// A chunked, append-only arena allocator for CFR nodes.
///
/// Nodes are stored in fixed-size chunks so that they never move once allocated.
/// This enables lock-free reads by index: once a node is written and visible
/// (via the `len` atomic), its memory location is stable forever.
///
/// - **Reads** (`get`, `iter`): Lock-free. Uses a pre-built atomic pointer table
///   (`chunk_ptrs`) to find chunk data without any lock. The caller observes
///   `len` with `Acquire` ordering, which synchronizes with the `Release` store
///   after the node is fully initialized.
/// - **Writes** (`push`): A `Mutex` serializes all pushes. The node index is
///   determined inside the mutex to prevent races.
///
/// # Safety
///
/// `get()` uses unsafe to dereference raw pointers into chunk data. This is
/// sound because:
/// 1. We only read indices `< len.load(Acquire)`.
/// 2. `len` is only incremented (with `Release`) after the node is fully written.
/// 3. Chunk pointers in `chunk_ptrs` are set with `Release` before `len` advances.
/// 4. Nodes never move (chunks are `Box`ed and never reallocated/removed).
pub struct NodeArena {
    /// Heap-owned chunks. New chunks are appended under the mutex but never
    /// removed or moved.
    chunks: Mutex<Vec<Box<[MaybeUninit<Node>; CHUNK_SIZE]>>>,
    /// Lock-free table of raw pointers to chunk data.
    /// `chunk_ptrs[i]` points to the start of chunk `i`'s array.
    /// Set atomically during `push()` when a new chunk is allocated.
    /// Readers load these pointers without any lock.
    chunk_ptrs: Box<[AtomicPtr<Node>]>,
    /// Number of nodes that have been fully initialized and are visible to readers.
    len: AtomicUsize,
}

impl NodeArena {
    /// Create a new empty arena.
    pub fn new() -> Self {
        // Allocate the pointer table on the heap via Vec to avoid stack overflow.
        let chunk_ptrs: Box<[AtomicPtr<Node>]> = (0..MAX_CHUNKS)
            .map(|_| AtomicPtr::new(ptr::null_mut()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        NodeArena {
            chunks: Mutex::new(Vec::new()),
            chunk_ptrs,
            len: AtomicUsize::new(0),
        }
    }

    /// Append a node to the arena and return its index.
    ///
    /// The node's `idx` field is set to the actual allocated index inside the
    /// mutex, so callers can pass any placeholder value for `node.idx`.
    ///
    /// This acquires the mutex to serialize all pushes. A new chunk is allocated
    /// only when the current chunk is full. The node is written to its slot,
    /// then `len` is incremented with `Release` ordering to make it visible
    /// to concurrent readers.
    pub fn push(&self, mut node: Node) -> usize {
        let mut chunks = self.chunks.lock();

        // Read len inside the mutex to prevent TOCTOU races between
        // concurrent pushers.
        let idx = self.len.load(Ordering::Relaxed);

        // Set the node's index to match its actual position in the arena.
        node.idx = idx as u32;

        let chunk_idx = idx / CHUNK_SIZE;
        let slot_idx = idx % CHUNK_SIZE;

        assert!(
            chunk_idx < MAX_CHUNKS,
            "NodeArena: exceeded maximum capacity of {} nodes",
            MAX_CHUNKS * CHUNK_SIZE
        );

        // Allocate new chunk if needed.
        while chunks.len() <= chunk_idx {
            let chunk = Box::new([const { MaybeUninit::uninit() }; CHUNK_SIZE]);
            // Store the raw pointer for lock-free reads.
            // Safety: The Box gives a stable heap pointer that remains valid
            // as long as the Box lives (until the arena is dropped).
            let chunk_ptr = chunk.as_ptr() as *mut Node;
            chunks.push(chunk);
            self.chunk_ptrs[chunks.len() - 1].store(chunk_ptr, Ordering::Release);
        }

        // Write the node into the slot while holding the lock.
        chunks[chunk_idx][slot_idx].write(node);

        // Make the node visible to readers.
        // Release ordering ensures all writes above are visible before len advances.
        self.len.store(idx + 1, Ordering::Release);

        idx
    }

    /// Get a reference to the node at `idx`.
    ///
    /// This is **lock-free**: it uses an atomic load on `len` and a raw pointer
    /// lookup into the chunk pointer table. No mutex is acquired.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len()`.
    ///
    /// # Safety rationale
    ///
    /// The unsafe block dereferences a raw pointer into chunk data. This is
    /// sound because:
    /// - We check `idx < len.load(Acquire)`, synchronizing with `push()`'s
    ///   `Release` store on `len`.
    /// - The node at `idx` was fully written before `len` was incremented.
    /// - The chunk pointer was stored with `Release` before `len` advanced.
    /// - Chunk memory is heap-allocated (`Box`) and never freed while the
    ///   arena lives.
    pub fn get(&self, idx: usize) -> &Node {
        let current_len = self.len.load(Ordering::Acquire);
        assert!(
            idx < current_len,
            "NodeArena::get({idx}) out of bounds (len={current_len})"
        );

        let chunk_idx = idx / CHUNK_SIZE;
        let slot_idx = idx % CHUNK_SIZE;

        // Load the chunk pointer lock-free. The Acquire on `len` above already
        // provides the necessary synchronization with push()'s Release store.
        let chunk_ptr = self.chunk_ptrs[chunk_idx].load(Ordering::Acquire);
        debug_assert!(!chunk_ptr.is_null());

        // Safety: The node at this slot was initialized before len was
        // incremented past idx. The chunk is heap-allocated and never moves.
        unsafe { &*chunk_ptr.add(slot_idx) }
    }

    /// Return the number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Iterate over all nodes in the arena.
    ///
    /// The iterator captures the current length at creation time,
    /// so nodes added during iteration are not visited.
    pub fn iter(&self) -> NodeArenaIter<'_> {
        NodeArenaIter {
            arena: self,
            current: 0,
            len: self.len(),
        }
    }
}

impl Drop for NodeArena {
    fn drop(&mut self) {
        // Drop all initialized nodes. MaybeUninit doesn't call drop on its
        // contents, so we must do it manually for slots that were initialized.
        let len = *self.len.get_mut();
        let chunks = self.chunks.get_mut();
        for i in 0..len {
            let chunk_idx = i / CHUNK_SIZE;
            let slot_idx = i % CHUNK_SIZE;
            // Safety: All slots 0..len were initialized via push().
            unsafe {
                chunks[chunk_idx][slot_idx].assume_init_drop();
            }
        }
    }
}

impl std::fmt::Debug for NodeArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeArena")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// Iterator over all nodes in a `NodeArena`.
pub struct NodeArenaIter<'a> {
    arena: &'a NodeArena,
    current: usize,
    len: usize,
}

impl<'a> Iterator for NodeArenaIter<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.len {
            let node = self.arena.get(self.current);
            self.current += 1;
            Some(node)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NodeArenaIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::cfr::NodeData;

    #[test]
    fn test_push_and_get() {
        let arena = NodeArena::new();
        let root = Node::new_root();
        let idx = arena.push(root);
        assert_eq!(idx, 0);
        assert_eq!(arena.len(), 1);

        let node = arena.get(0);
        assert_eq!(node.idx, 0);
    }

    #[test]
    fn test_multiple_pushes() {
        let arena = NodeArena::new();
        for i in 0..100 {
            let node = Node::new(0, 0, NodeData::Chance);
            let idx = arena.push(node);
            assert_eq!(idx, i);
        }
        assert_eq!(arena.len(), 100);

        for i in 0..100 {
            let node = arena.get(i);
            assert_eq!(node.idx, i as u32);
        }
    }

    #[test]
    fn test_cross_chunk_boundary() {
        let arena = NodeArena::new();
        // Push more than one chunk's worth of nodes
        for _ in 0..(CHUNK_SIZE + 10) {
            let node = Node::new(0, 0, NodeData::Chance);
            arena.push(node);
        }
        assert_eq!(arena.len(), CHUNK_SIZE + 10);

        // Verify nodes across the chunk boundary
        let last_in_first_chunk = arena.get(CHUNK_SIZE - 1);
        assert_eq!(last_in_first_chunk.idx, (CHUNK_SIZE - 1) as u32);

        let first_in_second_chunk = arena.get(CHUNK_SIZE);
        assert_eq!(first_in_second_chunk.idx, CHUNK_SIZE as u32);
    }

    #[test]
    fn test_iter() {
        let arena = NodeArena::new();
        for _ in 0..5 {
            let node = Node::new(0, 0, NodeData::Chance);
            arena.push(node);
        }

        let indices: Vec<u32> = arena.iter().map(|n| n.idx).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let arena = NodeArena::new();
        arena.get(0);
    }

    #[test]
    fn test_concurrent_reads() {
        use std::sync::Arc;

        let arena = Arc::new(NodeArena::new());
        for _ in 0..100 {
            let node = Node::new(0, 0, NodeData::Chance);
            arena.push(node);
        }

        // Spawn threads that read concurrently
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let arena = arena.clone();
                std::thread::spawn(move || {
                    for i in 0..100 {
                        let node = arena.get(i);
                        assert_eq!(node.idx, i as u32);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_push_and_read() {
        use std::sync::Arc;

        let arena = Arc::new(NodeArena::new());
        // Pre-populate some nodes
        for i in 0..50 {
            let node = Node::new(0, 0, NodeData::Chance);
            let idx = arena.push(node);
            assert_eq!(idx, i);
        }

        let arena_writer = arena.clone();
        let writer = std::thread::spawn(move || {
            for _ in 50..100 {
                let node = Node::new(0, 0, NodeData::Chance);
                arena_writer.push(node);
            }
        });

        // Reader reads pre-populated nodes while writer pushes new ones
        for i in 0..50 {
            let node = arena.get(i);
            assert_eq!(node.idx, i as u32);
        }

        writer.join().unwrap();
        assert_eq!(arena.len(), 100);
    }

    /// Compile-time assertion that NodeArena is Send + Sync.
    /// This is critical for correctness: NodeArena is shared across threads
    /// via Arc<NodeArena> in CFRState, so it must be safe to send and share.
    #[test]
    fn test_node_arena_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NodeArena>();
    }

    #[test]
    fn test_iter_snapshot_ignores_concurrent_pushes() {
        let arena = NodeArena::new();
        for _ in 0..5 {
            arena.push(Node::new(0, 0, NodeData::Chance));
        }

        let iter = arena.iter();
        // Push more after creating the iterator
        arena.push(Node::new(0, 0, NodeData::Chance));
        arena.push(Node::new(0, 0, NodeData::Chance));

        // Iterator should only see the first 5
        assert_eq!(iter.len(), 5);
        assert_eq!(iter.count(), 5);
        assert_eq!(arena.len(), 7);
    }

    #[test]
    fn test_concurrent_pushes() {
        use std::sync::Arc;

        let arena = Arc::new(NodeArena::new());
        let num_threads = 4;
        let pushes_per_thread = 250;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let arena = arena.clone();
                std::thread::spawn(move || {
                    for _ in 0..pushes_per_thread {
                        let node = Node::new(0, 0, NodeData::Chance);
                        arena.push(node);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(arena.len(), num_threads * pushes_per_thread);

        // All nodes should be readable and have unique indices
        let mut indices: Vec<u32> = arena.iter().map(|n| n.idx).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), num_threads * pushes_per_thread);
    }
}
