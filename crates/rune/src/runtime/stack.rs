use core::array;
use core::fmt;
use core::mem::replace;
use core::slice;

use crate::alloc::alloc::Global;
use crate::alloc::prelude::*;
use crate::alloc::{self, Vec};
use crate::runtime::{InstAddress, Value, VmErrorKind};

/// An error raised when interacting with the stack.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub struct StackError;

impl fmt::Display for StackError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tried to access out-of-bounds stack entry")
    }
}

cfg_std! {
    impl std::error::Error for StackError {}
}

/// The stack of the virtual machine, where all values are stored.
#[derive(Default, Debug)]
pub struct Stack {
    /// The current stack of values.
    stack: Vec<Value>,
    /// The top of the current stack frame.
    ///
    /// It is not possible to interact with values below this stack frame.
    stack_bottom: usize,
}

impl Stack {
    /// Construct a new stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(rune::to_value(String::from("Hello World"))?);
    /// let value = stack.pop()?;
    /// let value: String = rune::from_value(value)?;
    /// assert_eq!(value, "Hello World");
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            stack: Vec::new(),
            stack_bottom: 0,
        }
    }

    /// The current top address of the stack.
    #[inline]
    pub const fn addr(&self) -> InstAddress {
        InstAddress::new(self.stack.len().saturating_sub(self.stack_bottom))
    }

    /// Try to resize the stack with space for the given size.
    pub(crate) fn resize(&mut self, size: usize) -> alloc::Result<()> {
        if size == 0 {
            return Ok(());
        }

        let empty = Value::empty()?;
        self.stack.try_resize(self.stack_bottom + size, empty)?;
        Ok(())
    }

    /// Construct a new stack with the given capacity pre-allocated.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// let mut stack = Stack::with_capacity(16)?;
    /// assert!(stack.pop().is_err());
    /// stack.push(rune::to_value(String::from("Hello World"))?);
    /// let value = stack.pop()?;
    /// let value: String = rune::from_value(value)?;
    /// assert_eq!(value, "Hello World");
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    pub fn with_capacity(capacity: usize) -> alloc::Result<Self> {
        Ok(Self {
            stack: Vec::try_with_capacity(capacity)?,
            stack_bottom: 0,
        })
    }

    /// Check if the stack is empty.
    ///
    /// This ignores [stack_bottom] and will just check if the full stack is
    /// empty.
    ///
    /// ```
    /// use rune::runtime::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert!(stack.is_empty());
    /// stack.push(rune::to_value(String::from("Hello World"))?);
    /// assert!(!stack.is_empty());
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Get the length of the stack.
    ///
    /// This ignores [stack_bottom] and will just return the total length of
    /// the stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert_eq!(stack.len(), 0);
    /// stack.push(rune::to_value(String::from("Hello World"))?);
    /// assert_eq!(stack.len(), 1);
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Perform a raw access over the stack.
    ///
    /// This ignores [stack_bottom] and will just check that the given slice
    /// index is within range.
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn get<I>(&self, index: I) -> Option<&<I as slice::SliceIndex<[Value]>>::Output>
    where
        I: slice::SliceIndex<[Value]>,
    {
        self.stack.get(index)
    }

    /// Push a value onto the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(rune::to_value(String::from("Hello World"))?);
    /// assert_eq!(rune::from_value::<String>(stack.pop()?)?, "Hello World");
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    pub fn push<T>(&mut self, value: T) -> alloc::Result<()>
    where
        T: TryInto<Value, Error: Into<alloc::Error>>,
    {
        self.stack.try_push(value.try_into().map_err(Into::into)?)?;
        Ok(())
    }

    /// Drain the current stack down to the current stack bottom.
    pub(crate) fn drain(&mut self) -> impl DoubleEndedIterator<Item = Value> + '_ {
        self.stack.drain(self.stack_bottom..)
    }

    /// Get the slice at the given address with the given length.
    ///
    /// # Examples
    ///
    /// ```
    /// use rune::runtime::{Stack, InstAddress};
    ///
    /// let mut stack = Stack::new();
    /// stack.push(rune::to_value(1i64)?);
    /// stack.push(rune::to_value(1i64)?);
    /// stack.push(rune::to_value(1i64)?);
    ///
    /// let values = stack.slice_at(InstAddress::ZERO, 2)?;
    /// assert_eq!(values.len(), 2);
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    pub fn slice_at(&self, addr: InstAddress, count: usize) -> Result<&[Value], StackError> {
        let Some(start) = self.stack_bottom.checked_add(addr.offset()) else {
            return Err(StackError);
        };

        let Some(end) = start.checked_add(count) else {
            return Err(StackError);
        };

        let Some(slice) = self.stack.get(start..end) else {
            return Err(StackError);
        };

        Ok(slice)
    }

    /// Get the mutable slice at the given address with the given length.
    pub fn slice_at_mut(
        &mut self,
        addr: InstAddress,
        count: usize,
    ) -> Result<&mut [Value], StackError> {
        let Some(start) = self.stack_bottom.checked_add(addr.offset()) else {
            return Err(StackError);
        };

        let Some(end) = start.checked_add(count) else {
            return Err(StackError);
        };

        let Some(slice) = self.stack.get_mut(start..end) else {
            return Err(StackError);
        };

        Ok(slice)
    }

    /// Get the slice at the given address with the given static length.
    pub fn array_at<const N: usize>(&self, addr: InstAddress) -> Result<[&Value; N], StackError> {
        let slice = self.slice_at(addr, N)?;
        Ok(array::from_fn(|i| &slice[i]))
    }

    /// Extend the current stack with an iterator.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::alloc::String;
    /// use rune::Value;
    ///
    /// let mut stack = Stack::new();
    ///
    /// stack.extend([
    ///     rune::to_value(42i64)?,
    ///     rune::to_value(String::try_from("foo")?)?,
    ///     rune::to_value(())?
    /// ]);
    ///
    /// let values = stack.drain(2)?.collect::<Vec<_>>();
    ///
    /// assert_eq!(values.len(), 2);
    /// assert_eq!(rune::from_value::<String>(&values[0])?, "foo");
    /// assert_eq!(rune::from_value(&values[1])?, ());
    /// # Ok::<_, rune::support::Error>(())
    /// ```
    pub fn extend<I>(&mut self, iter: I) -> alloc::Result<()>
    where
        I: IntoIterator<Item = Value>,
    {
        for value in iter {
            self.stack.try_push(value)?;
        }

        Ok(())
    }

    /// Clear the current stack.
    pub fn clear(&mut self) {
        self.stack.clear();
        self.stack_bottom = 0;
    }

    /// Get the last position on the stack.
    #[inline]
    pub fn last(&self) -> Result<&Value, StackError> {
        self.stack.last().ok_or(StackError)
    }

    /// Iterate over the stack.
    pub fn iter(&self) -> impl Iterator<Item = &Value> + '_ {
        self.stack.iter()
    }

    /// Get the offset that corresponds to the bottom of the stack right now.
    ///
    /// The stack is partitioned into call frames, and once we enter a call
    /// frame the bottom of the stack corresponds to the bottom of the current
    /// call frame.
    pub fn stack_bottom(&self) -> usize {
        self.stack_bottom
    }

    /// Access the value at the given frame offset.
    pub(crate) fn at(&self, addr: InstAddress) -> Result<&Value, StackError> {
        self.stack_bottom
            .checked_add(addr.offset())
            .and_then(|n| self.stack.get(n))
            .ok_or(StackError)
    }

    /// Get a value mutable at the given index from the stack bottom.
    pub(crate) fn at_mut(&mut self, addr: InstAddress) -> Result<&mut Value, StackError> {
        self.stack_bottom
            .checked_add(addr.offset())
            .and_then(|n| self.stack.get_mut(n))
            .ok_or(StackError)
    }

    /// Swap the value at position a with the value at position b.
    pub(crate) fn swap(&mut self, a: InstAddress, b: InstAddress) -> Result<(), StackError> {
        if a == b {
            return Ok(());
        }

        let a = self
            .stack_bottom
            .checked_add(a.offset())
            .filter(|&n| n < self.stack.len())
            .ok_or(StackError)?;

        let b = self
            .stack_bottom
            .checked_add(b.offset())
            .filter(|&n| n < self.stack.len())
            .ok_or(StackError)?;

        self.stack.swap(a, b);
        Ok(())
    }

    /// Modify stack top by subtracting the given count from it while checking
    /// that it is in bounds of the stack.
    ///
    /// This is used internally when returning from a call frame.
    ///
    /// Returns the old stack top.
    #[tracing::instrument(skip_all)]
    pub(crate) fn swap_stack_bottom(
        &mut self,
        addr: InstAddress,
        len: usize,
    ) -> Result<usize, VmErrorKind> {
        let Some(start) = self.stack_bottom.checked_add(addr.offset()) else {
            return Err(VmErrorKind::StackError);
        };

        let old_len = self.stack.len();

        let Some(new_len) = old_len.checked_add(len) else {
            return Err(VmErrorKind::StackError);
        };

        if old_len < start + len {
            return Err(VmErrorKind::StackError);
        }

        self.stack.try_reserve(len)?;

        // SAFETY: We've ensured that the collection has space for the new
        // values. It is also guaranteed to be non-overlapping.
        unsafe {
            let ptr = self.stack.as_mut_ptr();
            let from = slice::from_raw_parts(ptr.add(start), len);

            for (value, n) in from.iter().zip(old_len..) {
                ptr.add(n).write(value.clone());
            }

            self.stack.set_len(new_len);
        }

        Ok(replace(&mut self.stack_bottom, old_len))
    }

    /// Pop the current stack top and modify it to a different one.
    ///
    /// This asserts that the size of the current stack frame is exactly zero
    /// before restoring it.
    #[tracing::instrument(skip_all)]
    pub(crate) fn pop_stack_top(&mut self, stack_bottom: usize) -> alloc::Result<()> {
        tracing::trace!(stack = self.stack.len(), self.stack_bottom);
        self.stack.truncate(self.stack_bottom);
        self.stack_bottom = stack_bottom;
        Ok(())
    }
}

impl TryClone for Stack {
    fn try_clone(&self) -> alloc::Result<Self> {
        Ok(Self {
            stack: self.stack.try_clone()?,
            stack_bottom: self.stack_bottom,
        })
    }
}

impl TryFromIteratorIn<Value, Global> for Stack {
    fn try_from_iter_in<T: IntoIterator<Item = Value>>(
        iter: T,
        alloc: Global,
    ) -> alloc::Result<Self> {
        Ok(Self {
            stack: iter.into_iter().try_collect_in(alloc)?,
            stack_bottom: 0,
        })
    }
}

impl From<Vec<Value>> for Stack {
    fn from(stack: Vec<Value>) -> Self {
        Self {
            stack,
            stack_bottom: 0,
        }
    }
}
